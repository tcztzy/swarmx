import { randomUUID } from "node:crypto";
import { chmodSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { EventType } from "@ag-ui/core";
import { z } from "zod";
import type { EventAttributes } from "../agents/types.js";
import type { ExecutionRecord } from "../execution-record.js";
import {
  CreateWorkItemSchema,
  ReconcileWorkOutcomeSchema,
  ReviseWorkSchema,
  type SelectionEvidence,
  SetWorkBudgetSchema,
  WorkArtifactSchema,
  type WorkAttempt,
  WorkAttemptSchema,
  WorkChargeSchema,
  type WorkConfiguration,
  WorkConfigurationSchema,
  WorkCycleSchema,
  WorkDecisionSchema,
  WorkFeedbackInputSchema,
  WorkFeedbackSchema,
  type WorkItem,
  WorkItemSchema,
  type WorkRunSchema,
  WorkSnapshotSchema,
} from "../work.js";
import { executionRuns, executionToolCost } from "./execution-evidence.js";
import type { ExecutionJournal } from "./execution-journal.js";

const usd = (amount: number) => Number(amount.toFixed(12));
const State = z.strictObject({
  version: z.literal(1),
  cycles: z.array(WorkCycleSchema),
  items: z.array(WorkItemSchema),
  reservations: z.array(
    WorkAttemptSchema.extend({
      costUsd: WorkAttemptSchema.shape.costUsd.default(null),
      costSource: WorkAttemptSchema.shape.costSource.default("unknown"),
      coverage: WorkAttemptSchema.shape.coverage.default("unknown"),
    }),
  ),
  feedback: z.array(WorkFeedbackSchema),
  decisions: z.array(WorkDecisionSchema),
  charges: z.array(WorkChargeSchema),
});
type State = z.infer<typeof State>;

function chooseConfiguration(
  item: WorkItem,
  candidates: WorkConfiguration[],
  evidence: SelectionEvidence[],
) {
  if (item.policy === "fixed") return candidates.find(({ id }) => id === item.fixedConfiguration);
  const cost = (id: string) =>
    evidence.find((row) => row.configurationId === id)?.observedCostUsd ?? Infinity;
  const ordered = [...candidates].sort(
    (a, b) => cost(a.id) - cost(b.id) || a.id.localeCompare(b.id),
  );
  if (item.policy === "adaptive") {
    const accepted = ordered.filter((candidate) =>
      evidence.some((row) => row.configurationId === candidate.id && row.accepted > 0),
    );
    if (accepted.length) return accepted[0];
    return (
      ordered.find(
        (candidate) =>
          !evidence.some((row) => row.configurationId === candidate.id && row.samples > 0),
      ) ?? ordered[0]
    );
  }
  return ordered[0];
}

/** Host-owned admission and acceptance; scientific bytes and native transcripts stay elsewhere. */
export class WorkManager {
  private readonly database: DatabaseSync;
  constructor(
    root: string,
    private readonly directoryKey: string,
    private readonly journal: ExecutionJournal,
    private readonly feedbackChanged: () => void = () => {},
  ) {
    mkdirSync(root, { recursive: true, mode: 0o700 });
    const path = join(root, "work.sqlite");
    this.database = new DatabaseSync(path);
    chmodSync(path, 0o600);
    this.database.exec(`PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS work_state (workspace TEXT PRIMARY KEY, state TEXT NOT NULL CHECK(json_valid(state))) STRICT;`);
    this.database.prepare("INSERT OR IGNORE INTO work_state VALUES (?, ?)").run(
      directoryKey,
      JSON.stringify({
        version: 1,
        cycles: [],
        items: [],
        reservations: [],
        feedback: [],
        decisions: [],
        charges: [],
      }),
    );
    this.flushFeedback(false);
  }

  private read(): State {
    const row = this.database
      .prepare("SELECT state FROM work_state WHERE workspace=?")
      .get(this.directoryKey) as { state: string };
    const state = State.parse(JSON.parse(row.state));
    const records = this.records();
    const runs = executionRuns(records);
    for (const reservation of state.reservations) {
      const invoice = state.charges.findLast((row) => row.reservationId === reservation.id);
      const owned = records.filter(
        (record) => record.attributes["swarmx.work.reservation_id"] === reservation.id,
      );
      const ids = new Set(owned.map((record) => record.runId));
      const observed = runs.filter((run) => ids.has(run.runId));
      reservation.runIds = observed.map((run) => run.runId);
      if (invoice) {
        reservation.costUsd = invoice.costUsd;
        reservation.costSource = "invoice";
        reservation.coverage = "complete";
      } else if (observed.length && observed.every((run) => run.costUsd !== null)) {
        reservation.costUsd = usd(
          observed.reduce((sum, run) => sum + (run.costUsd ?? 0), executionToolCost(owned).usd),
        );
        reservation.costSource = observed.every((run) => run.costSource === "invoice")
          ? "invoice"
          : "native-estimate";
        reservation.coverage = observed.every((run) => run.usageCoverage === "complete")
          ? "complete"
          : "partial";
      }
    }
    return state;
  }

  private change<T>(mutate: (state: State) => T): T {
    this.database.exec("BEGIN IMMEDIATE");
    try {
      const state = this.read();
      const result = mutate(state);
      this.database.prepare("UPDATE work_state SET state=? WHERE workspace=?").run(
        JSON.stringify({
          ...State.parse(state),
          reservations: state.reservations.map(
            ({ costUsd: _cost, costSource: _source, coverage: _coverage, ...row }) => row,
          ),
        }),
        this.directoryKey,
      );
      this.database.exec("COMMIT");
      return structuredClone(result);
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }

  private trusted() {
    if (this.journal.scope.getStore()?.sessionId)
      throw new Error("Work administration and acceptance require a trusted Host caller.");
  }

  createCycle(raw: unknown) {
    this.trusted();
    const input = WorkCycleSchema.parse(raw);
    if (new Set(input.configurations.map(({ id }) => id)).size !== input.configurations.length)
      throw new Error("Configuration IDs must be unique.");
    return this.change((state) => {
      if (state.cycles.some(({ id }) => id === input.id))
        throw new Error("Work cycle already exists.");
      state.cycles.push(input);
      return input;
    });
  }

  cycles() {
    return this.read().cycles;
  }

  setBudget(raw: unknown) {
    this.trusted();
    const input = SetWorkBudgetSchema.parse(raw);
    this.reconcile();
    return this.change((state) => {
      const cycle = state.cycles.find(({ id }) => id === input.cycleId);
      if (!cycle || cycle.budgetUsd !== input.expectedBudgetUsd)
        throw new Error("Work budget changed. Refresh before editing it.");
      if (state.reservations.some((row) => row.cycleId === cycle.id && row.finishedAt === null))
        throw new Error("Stop or reconcile unfinished executions before changing the budget.");
      const balance = this.balance(state, cycle.id);
      if (input.budgetUsd < usd(balance.spentUsd + balance.heldUsd))
        throw new Error("Budget cannot be lower than spent and reserved funds.");
      cycle.budgetUsd = input.budgetUsd;
      return cycle;
    });
  }

  createItem(raw: unknown) {
    this.trusted();
    const input = CreateWorkItemSchema.parse(raw);
    return this.change((state) => {
      const cycle = state.cycles.find(({ id }) => id === input.cycleId);
      if (!cycle) throw new Error("Unknown work cycle.");
      if (state.items.some(({ id }) => id === input.id))
        throw new Error("Work item already exists.");
      if (
        input.policy === "fixed" &&
        !cycle.configurations.some(({ id }) => id === input.fixedConfiguration)
      )
        throw new Error("Fixed policy requires a registered configuration.");
      for (const dependency of input.dependencies)
        if (!state.items.some(({ id, cycleId }) => id === dependency && cycleId === input.cycleId))
          throw new Error("Dependencies must be earlier items in the same cycle.");
      const item: WorkItem = {
        ...input,
        createdAt: new Date().toISOString(),
        state: "queued",
        blockedReason: null,
        firstAcceptedAt: null,
      };
      state.items.push(item);
      return item;
    });
  }

  records() {
    const records = [];
    let after = 0;
    for (;;) {
      const page = this.journal.read({ after, limit: 1000 });
      records.push(...page.events);
      if (page.events.length < 1000) return records;
      after = page.nextAfter;
    }
  }

  reconcile() {
    const records = this.records();
    const runs = executionRuns(records);
    const active = new Set(this.journal.activeRuns().map(({ runId }) => runId));
    this.change((state) => {
      for (const reservation of state.reservations) {
        const ids = new Set(
          records
            .filter((record) => record.attributes["swarmx.work.reservation_id"] === reservation.id)
            .map(({ runId }) => runId),
        );
        const observed = runs.filter(({ runId }) => ids.has(runId));
        reservation.runIds = observed.map(({ runId }) => runId);
        if (!observed.length) continue;
        const confirmed = reservation.outcome?.startsWith("confirmed-");
        const live = observed.some(({ runId }) => active.has(runId));
        const complete = observed.every(({ outcome }) => outcome !== "incomplete");
        if (!confirmed) {
          reservation.outcome = observed.map(({ outcome }) => outcome).join(",");
          reservation.finishedAt = complete ? (observed.at(-1)?.finishedAt ?? null) : null;
        }
        reservation.state = live
          ? "running"
          : reservation.finishedAt !== null && reservation.costUsd !== null
            ? "settled"
            : "uncertain";
        const item = state.items.find(({ id }) => id === reservation.workId);
        const latest = state.reservations.findLast(
          (row) => row.workId === reservation.workId && row.purpose === "execution",
        );
        if (item && ["running", "blocked"].includes(item.state) && latest?.id === reservation.id) {
          if (!live && reservation.finishedAt !== null) {
            item.state = "awaiting-acceptance";
            item.blockedReason = null;
          } else if (reservation.state === "uncertain") {
            item.state = "blocked";
            item.blockedReason =
              "Execution outcome is unknown; reconcile before explicit continuation.";
          }
        }
      }
    });
  }

  private balance(state: State, cycleId: string) {
    const cycle = state.cycles.find(({ id }) => id === cycleId);
    if (!cycle) throw new Error("Unknown work cycle.");
    const reservations = state.reservations.filter((row) => row.cycleId === cycleId);
    const records = this.cycleRecords(state, cycleId);
    const spentUsd = this.spent(reservations, records);
    const heldUsd = usd(
      reservations.reduce((sum, root) => {
        if (root.runtimeId !== root.id) return sum;
        const group = reservations.filter((row) => row.runtimeId === root.id);
        return (
          sum +
          (group.some((row) => row.costUsd === null || row.finishedAt === null)
            ? Math.max(
                0,
                root.reservedUsd -
                  this.spent(
                    group,
                    records.filter(
                      (record) => record.attributes["swarmx.work.runtime_id"] === root.id,
                    ),
                  ),
              )
            : 0)
        );
      }, 0),
    );
    return {
      budgetUsd: cycle.budgetUsd,
      spentUsd,
      heldUsd,
      remainingUsd: usd(cycle.budgetUsd - spentUsd - heldUsd),
      costCoverage: {
        reported: reservations.filter((row) => row.costUsd !== null).length,
        total: reservations.length,
        completeExecutions: reservations.filter((row) => row.coverage === "complete").length,
      },
      enforcement:
        "Host admission reservations; native internal spending is not universally capped",
    };
  }

  private cycleRecords(state: State, cycleId: string) {
    const items = new Set(
      state.items.filter((item) => item.cycleId === cycleId).map((item) => item.id),
    );
    return this.records().filter(
      (record) =>
        record.attributes["swarmx.work.cycle_id"] === cycleId ||
        (typeof record.attributes["swarmx.work.item_id"] === "string" &&
          items.has(record.attributes["swarmx.work.item_id"])),
    );
  }

  private spent(reservations: WorkAttempt[], records: ExecutionRecord[]) {
    const invoices = new Map(
      reservations
        .filter((row) => row.costSource === "invoice")
        .map((row) => [row.id, row.costUsd ?? 0]),
    );
    const unbilled = records.filter((record) => {
      const id = record.attributes["swarmx.work.reservation_id"];
      return typeof id !== "string" || !invoices.has(id);
    });
    return usd(
      [...invoices.values()].reduce((sum, cost) => sum + cost, 0) +
        executionRuns(unbilled).reduce((sum, run) => sum + (run.costUsd ?? 0), 0) +
        executionToolCost(unbilled).usd,
    );
  }

  snapshot(cycleId: string) {
    this.reconcile();
    const state = this.read();
    const items = state.items.filter((row) => row.cycleId === cycleId);
    const reservations = state.reservations.filter((row) => row.cycleId === cycleId);
    const ids = new Set(reservations.map(({ id }) => id));
    const runIds = new Set(reservations.flatMap(({ runIds }) => runIds));
    const feedback = state.feedback.filter((row) => ids.has(row.attemptId));
    const values = items.map((item) => {
      const attempt = reservations.findLast(
        (row) =>
          row.workId === item.id &&
          row.purpose === "execution" &&
          row.criteriaVersion === item.criteriaVersion,
      );
      const latest = feedback.findLast(
        (row) => row.attemptId === attempt?.id && row.layer !== "structure",
      );
      return { item, latest };
    });
    return WorkSnapshotSchema.parse({
      cycle: state.cycles.find(({ id }) => id === cycleId),
      items,
      reservations,
      feedback,
      executions: executionRuns(this.records()).filter(({ runId }) => runIds.has(runId)),
      outcomes: {
        acceptedItems: items.filter(({ state }) => state === "accepted").length,
        acceptedValue: values.reduce(
          (sum, { item, latest }) => sum + (latest?.accepted ? item.value * latest.fraction : 0),
          0,
        ),
        partialValue: values.reduce(
          (sum, { item, latest }) =>
            sum + (latest && !latest.accepted ? item.value * latest.fraction : 0),
          0,
        ),
        interventions: feedback.filter(({ intervention }) => intervention !== "none").length,
        valueBasis:
          "User-configured work value; no implied conversion to cash or measured research benefit",
      },
      decisions: state.decisions.filter((row) => items.some(({ id }) => id === row.workId)),
      tools: executionToolCost(this.cycleRecords(state, cycleId)),
      balance: this.balance(state, cycleId),
      coverageGaps: [
        "Unreported native internal activity",
        "Tool/service cash without an invoice",
        "Preparation, catalog and retrieval resource usage beyond observed wall time",
        "Subscription quotas and local resource usage unless supplied by a trusted caller",
      ],
    });
  }

  item(id: string) {
    const item = this.read().items.find((row) => row.id === id);
    if (!item) throw new Error("Unknown work item.");
    return item;
  }

  attempt(id: string) {
    const attempt = this.read().reservations.find((row) => row.id === id);
    if (!attempt) throw new Error("Unknown work attempt.");
    return attempt;
  }

  revise(raw: unknown) {
    this.trusted();
    const input = ReviseWorkSchema.parse(raw);
    this.reconcile();
    return this.change((state) => {
      const item = state.items.find(({ id }) => id === input.id);
      if (!item || item.criteriaVersion !== input.expectedCriteriaVersion)
        throw new Error("Acceptance criteria revision changed.");
      if (
        state.reservations.some((row) => row.workId === item.id && row.finishedAt === null) ||
        input.criteriaVersion === item.criteriaVersion
      )
        throw new Error("Stop running work and provide a new criteria version.");
      Object.assign(item, {
        goal: input.goal,
        criteria: input.criteria,
        criteriaVersion: input.criteriaVersion,
        state: "queued",
        blockedReason: null,
      });
      return item;
    });
  }

  assertSession(sessionId: string) {
    const previous = this.records().find(
      (record) =>
        record.sessionId === sessionId &&
        typeof record.attributes["swarmx.work.item_id"] === "string",
    );
    if (
      previous &&
      previous.attributes["swarmx.work.item_id"] !==
        this.journal.scope.getStore()?.attributes["swarmx.work.item_id"]
    )
      throw new Error("A managed session must retain its original work identity and budget.");
  }

  private evidence(
    state: State,
    item: WorkItem,
    candidates: WorkConfiguration[],
  ): SelectionEvidence[] {
    const records = this.records();
    const runs = executionRuns(records);
    return candidates.map((candidate) => {
      const attempts = state.reservations.filter((attempt) => {
        const prior = state.items.find(({ id }) => id === attempt.workId);
        return (
          prior?.cycleId === item.cycleId &&
          prior.taskClass === item.taskClass &&
          prior.risk === item.risk &&
          attempt.criteriaVersion === item.criteriaVersion &&
          attempt.purpose === "execution" &&
          attempt.configuration?.harness === candidate.harness &&
          attempt.configuration.model === candidate.model &&
          attempt.configuration.effort === candidate.effort &&
          attempt.configuration.profile === candidate.profile
        );
      });
      const feedback = attempts.flatMap((attempt) => {
        const latest = state.feedback.findLast(
          (row) => row.attemptId === attempt.id && row.layer !== "structure",
        );
        return latest ? [latest] : [];
      });
      const costs = attempts.flatMap((attempt) => {
        const group = state.reservations.filter((row) => row.runtimeId === attempt.id);
        return group.length && group.every((row) => row.costUsd !== null)
          ? [
              this.spent(
                group,
                records.filter(
                  (record) => record.attributes["swarmx.work.runtime_id"] === attempt.id,
                ),
              ),
            ]
          : [];
      });
      const elapsed = runs
        .filter(({ runId }) => attempts.some((attempt) => attempt.runIds.includes(runId)))
        .flatMap(({ elapsedMs }) => (elapsedMs === null ? [] : [elapsedMs]));
      return {
        configurationId: candidate.id,
        samples: feedback.length,
        accepted: feedback.filter((row) => row.accepted).length,
        meanFraction: feedback.length
          ? feedback.reduce((sum, row) => sum + row.fraction, 0) / feedback.length
          : null,
        observedCostUsd: costs.length ? costs.reduce((a, b) => a + b, 0) / costs.length : null,
        costSamples: costs.length,
        feedbackIds: feedback.map(({ id }) => id),
        latestFeedbackAt: feedback.at(-1)?.recordedAt ?? null,
        meanElapsedMs: elapsed.length ? elapsed.reduce((a, b) => a + b, 0) / elapsed.length : null,
        executionOutcomes: attempts.flatMap(({ outcome }) => (outcome ? [outcome] : [])),
      };
    });
  }

  prepare(workId: string, admitted: (candidate: WorkConfiguration) => boolean) {
    this.reconcile();
    const state = this.read();
    const item = this.item(workId);
    const cycle = state.cycles.find(({ id }) => id === item.cycleId);
    if (!cycle) throw new Error("Unknown work cycle.");
    const balance = this.balance(state, item.cycleId);
    const candidates = cycle.configurations.filter(admitted);
    return {
      item,
      ...balance,
      candidates,
      evidence: this.evidence(state, item, candidates),
      policyVersion: item.mode === "managed" ? "native-supervisor" : item.policy,
    };
  }

  reserve(
    workId: string,
    admitted: (candidate: WorkConfiguration) => boolean,
    child?: { harness: string; model?: string; effort?: string; profile?: string },
    options: z.infer<typeof WorkRunSchema> = {},
  ) {
    this.reconcile();
    return this.change((state) => {
      const item = state.items.find((row) => row.id === workId);
      if (!item) throw new Error("Unknown work item.");
      const cycle = state.cycles.find((row) => row.id === item.cycleId);
      if (!cycle) throw new Error("Unknown work cycle.");
      if (!child) {
        if (item.state === "accepted" || item.state === "running")
          throw new Error("Work is accepted or already running.");
        if (state.reservations.some((row) => row.workId === workId && row.state !== "settled"))
          throw new Error("Reconcile the unresolved attempt before continuing.");
        if (
          item.dependencies.some(
            (id) => !state.items.some((row) => row.id === id && row.state === "accepted"),
          )
        )
          throw new Error("Work dependencies are not accepted.");
      }
      const balance = this.balance(state, cycle.id);
      const candidates = cycle.configurations.filter(admitted);
      const evidence = this.evidence(state, item, candidates);
      const mode = options.mode ?? item.mode;
      const explicit = child
        ? WorkConfigurationSchema.parse(child)
        : mode === "managed"
          ? (options.supervisor ?? item.supervisor)
          : (options.configuration ?? item.configuration);
      if (mode === "managed" && !child && !explicit)
        throw new Error("Managed work requires an explicitly selected supervisor Agent.");
      const configuration = explicit ?? chooseConfiguration(item, candidates, evidence);
      if (configuration && !admitted(configuration))
        throw new Error("The selected Agent configuration is not permitted.");
      const runtime = options.runtime ?? item.runtime;
      const runtimeId = child
        ? this.journal.scope.getStore()?.attributes["swarmx.work.runtime_id"]
        : undefined;
      const owner =
        typeof runtimeId === "string"
          ? state.reservations.find((row) => row.id === runtimeId)
          : undefined;
      if (child && !owner) throw new Error("Delegation requires its active parent runtime.");
      const available = owner ? this.runtimeBalance(state, owner.id) : balance.remainingUsd;
      const amount = owner ? 0 : (runtime.budgetUsd ?? available);
      const expired = item.deadline !== undefined && Date.parse(item.deadline) <= Date.now();
      const concurrent = state.reservations.filter(
        (row) =>
          row.cycleId === cycle.id &&
          row.runtimeId === row.id &&
          state.reservations.some(
            (member) => member.runtimeId === row.id && member.finishedAt === null,
          ),
      ).length;
      const reason = expired
        ? "Work deadline has passed."
        : !owner && concurrent >= cycle.concurrency
          ? "Cycle concurrency is occupied."
          : !configuration
            ? "No admitted Agent configuration is available."
            : available <= 0 || amount > available
              ? "No remaining budget for this runtime."
              : explicit
                ? "Use the explicitly selected Agent configuration."
                : "Choose from prepared acceptance evidence and complete recursive execution costs.";
      const decision = {
        id: randomUUID(),
        workId,
        at: new Date().toISOString(),
        policyVersion: mode === "managed" ? "native-supervisor" : item.policy,
        remainingUsd: available,
        candidates,
        evidence,
        configurationId: configuration?.id ?? null,
        action: expired
          ? ("stop" as const)
          : (!owner && concurrent >= cycle.concurrency) ||
              !configuration ||
              available <= 0 ||
              amount > available
            ? ("defer" as const)
            : ("execute" as const),
        reason,
      };
      state.decisions.push(decision);
      if (decision.action !== "execute" || !configuration) {
        if (!child) {
          item.state = "blocked";
          item.blockedReason = reason;
        }
        return { decision, reservation: null };
      }
      const reservation = this.addReservation(
        state,
        item,
        amount,
        child ? "delegation" : "execution",
        configuration,
        owner?.id,
      );
      if (!child) {
        item.state = "running";
        item.blockedReason = null;
      }
      return { decision, reservation };
    });
  }

  private runtimeBalance(state: State, runtimeId: string) {
    const root = state.reservations.find((row) => row.id === runtimeId);
    if (!root) throw new Error("Unknown work runtime.");
    const group = state.reservations.filter((row) => row.runtimeId === runtimeId);
    if (group.some((row) => row.id !== root.id && row.finishedAt !== null && row.costUsd === null))
      throw new Error(
        "Finished child cost is unknown; reconcile its bill before delegating again.",
      );
    return usd(
      root.reservedUsd -
        this.spent(
          group,
          this.records().filter(
            (record) => record.attributes["swarmx.work.runtime_id"] === runtimeId,
          ),
        ),
    );
  }

  remainingRuntime(runtimeId: string) {
    return this.runtimeBalance(this.read(), runtimeId);
  }

  private addReservation(
    state: State,
    item: WorkItem,
    amount: number,
    purpose: WorkAttempt["purpose"],
    configuration: WorkConfiguration | null,
    runtimeId?: string,
  ) {
    const id = randomUUID();
    const reservation: WorkAttempt = {
      id,
      runtimeId: runtimeId ?? id,
      workId: item.id,
      cycleId: item.cycleId,
      configuration,
      purpose,
      reservedUsd: amount,
      costUsd: null,
      costSource: "unknown",
      coverage: "unknown",
      state: "reserved",
      outcome: null,
      createdAt: new Date().toISOString(),
      finishedAt: null,
      runIds: [],
      artifacts: [],
      criteriaVersion: item.criteriaVersion,
      report: null,
    };
    state.reservations.push(reservation);
    return reservation;
  }

  reserveReview(sourceRunIds: string[]) {
    this.reconcile();
    return this.change((state) => {
      const source = state.reservations.find((row) =>
        row.runIds.some((id) => sourceRunIds.includes(id)),
      );
      if (!source) return null;
      const item = state.items.find(({ id }) => id === source.workId);
      const cycle = state.cycles.find(({ id }) => id === source.cycleId);
      if (!item || !cycle) throw new Error("Review source work is missing.");
      if (
        !cycle.reviewReserveUsd ||
        cycle.reviewReserveUsd > this.balance(state, cycle.id).remainingUsd
      )
        throw new Error("Background learning needs an explicit affordable review reservation.");
      if (
        state.reservations.filter(
          (row) =>
            row.cycleId === cycle.id &&
            (["reserved", "running"].includes(row.state) || row.finishedAt === null),
        ).length >= cycle.concurrency
      )
        throw new Error("Foreground work occupies the cycle's execution slots.");
      return this.addReservation(state, item, cycle.reviewReserveUsd, "memory-review", null);
    });
  }

  attributes(reservation: WorkAttempt): EventAttributes {
    return {
      "swarmx.work.item_id": reservation.workId,
      "swarmx.work.cycle_id": reservation.cycleId,
      "swarmx.work.reservation_id": reservation.id,
      "swarmx.work.runtime_id": reservation.runtimeId,
      "swarmx.work.reserved_usd": this.remainingRuntime(reservation.runtimeId),
      "swarmx.work.configuration_id": reservation.configuration?.id ?? null,
      "swarmx.execution.purpose": reservation.purpose,
    };
  }

  finish(id: string, error?: unknown) {
    this.reconcile();
    const records = this.records().filter(
      (record) => record.attributes["swarmx.work.reservation_id"] === id,
    );
    const sends = new Map(
      records.flatMap(({ event, causedBy, attributes }) =>
        event.type === EventType.TOOL_CALL_ARGS &&
        typeof attributes["swarmx.actor.id"] === "string" &&
        typeof causedBy === "string" &&
        records.some(
          (record) =>
            record.id === causedBy &&
            record.event.type === EventType.TOOL_CALL_START &&
            record.event.toolCallName === "swarm" &&
            record.event.toolCallId === event.toolCallId &&
            record.attributes["swarmx.actor.id"] === attributes["swarmx.actor.id"],
        ) &&
        z.object({ action: z.literal("send_message") }).safeParse(JSON.parse(event.delta)).success
          ? [[event.toolCallId, causedBy] as const]
          : [],
      ),
    );
    const cancelled = records.findLast(
      ({ event, causedBy, attributes }) =>
        event.type === EventType.TOOL_CALL_RESULT &&
        typeof attributes["swarmx.actor.id"] === "string" &&
        sends.get(event.toolCallId) === causedBy &&
        z
          .object({ result: z.object({ stopReason: z.literal("cancelled") }) })
          .safeParse(JSON.parse(event.content)).success,
    );
    this.change((state) => {
      const reservation = state.reservations.find((row) => row.id === id);
      if (!reservation) throw new Error("Unknown reservation.");
      const confirmed = reservation.outcome?.startsWith("confirmed-");
      if (!reservation.runIds.length && !confirmed) {
        reservation.state = cancelled && reservation.costUsd !== null ? "settled" : "uncertain";
        reservation.outcome = cancelled ? "cancelled-before-dispatch" : "dispatch-unconfirmed";
        reservation.finishedAt = cancelled?.observedAt ?? null;
      }
      if (error && !confirmed)
        reservation.report = error instanceof Error ? error.message : String(error);
      const item = state.items.find((row) => row.id === reservation.workId);
      if (item?.state === "running" && reservation.purpose === "execution") {
        const finished = reservation.finishedAt !== null && reservation.runIds.length > 0;
        item.state = finished ? "awaiting-acceptance" : "blocked";
        item.blockedReason = finished
          ? null
          : cancelled
            ? "Execution was cancelled before native dispatch."
            : "Dispatch not confirmed; reconcile before continuation.";
      }
    });
  }

  reconcileCharge(raw: unknown) {
    this.trusted();
    const input = WorkChargeSchema.parse(raw);
    this.reconcile();
    return this.change((state) => {
      const duplicate = state.charges.find(({ id }) => id === input.id);
      if (duplicate) {
        if (JSON.stringify(duplicate) !== JSON.stringify(input))
          throw new Error("WorkChargeSchema idempotency conflict.");
        return duplicate;
      }
      const reservation = state.reservations.find(({ id }) => id === input.reservationId);
      if (!reservation) throw new Error("Unknown reservation.");
      if (reservation.state === "reserved" || reservation.state === "running")
        throw new Error("Wait for execution before reconciling its bill.");
      reservation.costUsd = input.costUsd;
      reservation.costSource = input.source;
      reservation.coverage = "complete";
      reservation.state = reservation.finishedAt === null ? "uncertain" : "settled";
      state.charges.push(input);
      return input;
    });
  }

  reconcileOutcome(raw: unknown) {
    this.trusted();
    const input = ReconcileWorkOutcomeSchema.parse(raw);
    this.reconcile();
    return this.change((state) => {
      const reservation = state.reservations.find(({ id }) => id === input.reservationId);
      if (!reservation) throw new Error("Unknown reservation.");
      if (reservation.state === "reserved" || reservation.state === "running")
        throw new Error("Wait for execution to stop before confirming its outcome.");
      const outcome = `confirmed-${input.outcome}`;
      if (reservation.outcome?.startsWith("confirmed-")) {
        if (reservation.outcome !== outcome || reservation.report !== input.reference)
          throw new Error("Outcome confirmation idempotency conflict.");
        return reservation;
      }
      if (reservation.finishedAt !== null)
        throw new Error("Execution already has a recorded terminal outcome.");
      reservation.outcome = outcome;
      reservation.finishedAt = new Date().toISOString();
      reservation.report = input.reference;
      reservation.state = reservation.costUsd === null ? "uncertain" : "settled";
      const item = state.items.find(({ id }) => id === reservation.workId);
      const latest = state.reservations.findLast(
        (row) => row.workId === reservation.workId && row.purpose === "execution",
      );
      if (item && item.state === "blocked" && latest?.id === reservation.id) {
        item.state = "awaiting-acceptance";
        item.blockedReason = null;
      }
      return reservation;
    });
  }

  submit(attemptId: string, raw: unknown) {
    const artifacts = z.array(WorkArtifactSchema).max(100).parse(raw);
    return this.change((state) => {
      const attempt = state.reservations.find(({ id }) => id === attemptId);
      if (!attempt) throw new Error("Unknown work attempt.");
      if (state.feedback.some((row) => row.attemptId === attemptId))
        throw new Error("Accepted submission is immutable; create another attempt for revisions.");
      attempt.artifacts = artifacts;
      return attempt;
    });
  }

  accept(raw: unknown) {
    this.trusted();
    const input = WorkFeedbackInputSchema.parse(raw);
    if ((input.layer === "structure" || input.verdict === "failed") && input.accepted)
      throw new Error("Structure or failed verification cannot accept work.");
    if (input.verdict === "failed" && input.fraction !== 0)
      throw new Error("Failed work cannot earn value.");
    const feedback = this.change((state) => {
      const duplicate = state.feedback.find(({ id }) => id === input.id);
      if (duplicate) {
        const { recordedAt: _, ...original } = duplicate;
        if (JSON.stringify(original) !== JSON.stringify(input))
          throw new Error("WorkFeedbackSchema idempotency conflict.");
        return duplicate;
      }
      const attempt = state.reservations.find(({ id }) => id === input.attemptId);
      const item = state.items.find(({ id }) => id === attempt?.workId);
      if (!attempt || !item || attempt.purpose !== "execution")
        throw new Error("Unknown work attempt.");
      if (input.criteriaVersion !== attempt.criteriaVersion)
        throw new Error("Acceptance criteria revision changed.");
      if (!attempt.finishedAt) throw new Error("An unfinished execution cannot be accepted.");
      if (JSON.stringify(input.artifacts) !== JSON.stringify(attempt.artifacts))
        throw new Error("Acceptance must pin the submitted artifact revisions.");
      const previous = state.feedback.findLast(
        (row) => row.attemptId === input.attemptId && row.layer === input.layer,
      );
      if ((previous?.id ?? undefined) !== input.supersedes)
        throw new Error("WorkFeedbackSchema must explicitly supersede the previous result.");
      const feedback = { ...input, recordedAt: new Date().toISOString() };
      state.feedback.push(feedback);
      const latest = state.reservations.findLast(
        (row) => row.workId === item.id && row.purpose === "execution",
      );
      if (
        input.layer !== "structure" &&
        item.criteriaVersion === attempt.criteriaVersion &&
        latest?.id === attempt.id
      ) {
        item.state = input.accepted ? "accepted" : "awaiting-acceptance";
        item.blockedReason = input.accepted ? null : input.report;
        if (input.accepted) item.firstAcceptedAt ??= feedback.recordedAt;
      }
      return feedback;
    });
    this.flushFeedback();
    return feedback;
  }

  flushFeedback(notify = true) {
    this.trusted();
    const state = this.read();
    const records = this.records();
    const published = new Set(
      records.flatMap(({ event }) =>
        event.type === EventType.CUSTOM && event.name === "swarmx.work.feedback"
          ? [event.value.feedback.id]
          : [],
      ),
    );
    let changed = false;
    for (const feedback of state.feedback) {
      if (published.has(feedback.id)) continue;
      const attempt = state.reservations.find(({ id }) => id === feedback.attemptId);
      if (!attempt) throw new Error("WorkFeedbackSchema attempt is missing.");
      const started = records.find(
        ({ runId, event }) =>
          runId !== null && attempt.runIds.includes(runId) && event.type === EventType.RUN_STARTED,
      );
      this.journal.append(
        {
          sessionId: started?.sessionId ?? null,
          runId: started?.runId ?? `work-feedback:${attempt.id}`,
          causedBy: started?.id ?? null,
          attributes: started?.attributes ?? {
            ...this.attributes(attempt),
            "swarmx.memory.review_eligible": false,
          },
        },
        {
          type: EventType.CUSTOM,
          timestamp: Date.parse(feedback.recordedAt),
          name: "swarmx.work.feedback",
          value: {
            workId: attempt.workId,
            cycleId: attempt.cycleId,
            attemptId: attempt.id,
            feedback,
            runIds: attempt.runIds,
            configuration: attempt.configuration,
          },
        },
        {},
        `work-feedback:${feedback.id}`,
      );
      changed = true;
    }
    if (changed && notify) this.feedbackChanged();
  }

  pending(cycleId: string) {
    const { items, reservations } = this.snapshot(cycleId);
    return items
      .filter(
        (item) =>
          (item.state === "queued" || item.state === "blocked") &&
          !reservations.some(
            (row) =>
              row.workId === item.id && row.purpose === "execution" && row.state !== "settled",
          ),
      )
      .sort(
        (a, b) =>
          b.priority - a.priority ||
          (a.deadline ?? "z").localeCompare(b.deadline ?? "z") ||
          b.value - a.value ||
          a.createdAt.localeCompare(b.createdAt),
      );
  }

  close() {
    this.database.close();
  }
}
