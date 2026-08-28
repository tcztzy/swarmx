export type VerificationMode = "enforced" | "prompt-only";

interface ModelState {
  readonly currentAttempt: "a" | "b" | null;
  readonly authorizedAttempts: readonly ("a" | "b")[];
  readonly effectAttempt: "a" | "b" | null;
  readonly effectStatus: "none" | "started" | "uncertain" | "succeeded" | "observed" | "absent";
  readonly uncertainOccurred: boolean;
  readonly externalEffects: number;
  readonly duplicateEffects: number;
  readonly staleEffects: number;
  readonly admissions: number;
  readonly knowledgeFacts: number;
  readonly messageDeliveries: number;
  readonly duplicateMessages: number;
  readonly recoveryReplays: number;
  readonly coordinationWrites: number;
  readonly crashed: boolean;
}

export interface ModelCheckReport {
  readonly mode: VerificationMode;
  readonly statesExplored: number;
  readonly safetyViolationRate: number;
  readonly duplicateEffects: number;
  readonly recoveryReplays: number;
  readonly knowledgePollution: number;
  readonly coordinationWrites: number;
}

const INITIAL: ModelState = {
  currentAttempt: null,
  authorizedAttempts: [],
  effectAttempt: null,
  effectStatus: "none",
  uncertainOccurred: false,
  externalEffects: 0,
  duplicateEffects: 0,
  staleEffects: 0,
  admissions: 0,
  knowledgeFacts: 0,
  messageDeliveries: 0,
  duplicateMessages: 0,
  recoveryReplays: 0,
  coordinationWrites: 0,
  crashed: false,
};

function key(state: ModelState): string {
  return JSON.stringify(state);
}

function assign(state: ModelState, attempt: "a" | "b", mode: VerificationMode): ModelState {
  return {
    ...state,
    currentAttempt: attempt,
    authorizedAttempts:
      mode === "enforced" ? [attempt] : [...new Set([...state.authorizedAttempts, attempt])].sort(),
    coordinationWrites: state.coordinationWrites + 1,
    crashed: false,
  };
}

function transitions(state: ModelState, mode: VerificationMode): ModelState[] {
  const next: ModelState[] = [];
  const unresolved = state.effectStatus === "started" || state.effectStatus === "uncertain";
  if (!state.crashed && (mode === "prompt-only" || !unresolved)) {
    next.push(assign(state, "a", mode), assign(state, "b", mode));
  }
  for (const attempt of ["a", "b"] as const) {
    const authorized = state.authorizedAttempts.includes(attempt);
    if (!state.crashed && authorized && (mode === "prompt-only" || !unresolved)) {
      next.push({
        ...state,
        effectAttempt: attempt,
        effectStatus: "started",
        duplicateEffects:
          mode === "prompt-only" && unresolved && state.uncertainOccurred
            ? state.duplicateEffects + 1
            : state.duplicateEffects,
        staleEffects: state.staleEffects + (attempt === state.currentAttempt ? 0 : 1),
        coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
      });
    }
  }
  if (state.effectStatus === "started") {
    next.push({
      ...state,
      effectStatus: "succeeded",
      externalEffects: state.externalEffects + 1,
      coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
    });
    next.push({
      ...state,
      effectStatus: "uncertain",
      uncertainOccurred: true,
      externalEffects: state.externalEffects + 1,
      coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
    });
    next.push({
      ...state,
      effectStatus: "uncertain",
      uncertainOccurred: false,
      coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
    });
  }
  if (state.effectStatus === "uncertain") {
    if (mode === "prompt-only" || state.uncertainOccurred) {
      next.push({
        ...state,
        effectStatus: "observed",
        coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
      });
    }
    if (mode === "prompt-only" || !state.uncertainOccurred) {
      next.push({
        ...state,
        effectStatus: "absent",
        coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
      });
    }
    if (mode === "prompt-only") {
      next.push({
        ...state,
        externalEffects: state.externalEffects + 1,
        duplicateEffects: state.duplicateEffects + (state.uncertainOccurred ? 1 : 0),
      });
    }
  }
  next.push({
    ...state,
    crashed: true,
    currentAttempt: null,
    authorizedAttempts: mode === "enforced" ? [] : state.authorizedAttempts,
    effectStatus: state.effectStatus === "started" ? "uncertain" : state.effectStatus,
    coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 1 : 0),
  });
  if (state.crashed) {
    next.push({
      ...state,
      crashed: false,
      externalEffects:
        mode === "prompt-only" && state.effectStatus === "uncertain"
          ? state.externalEffects + 1
          : state.externalEffects,
      recoveryReplays:
        mode === "prompt-only" && state.effectStatus === "uncertain"
          ? state.recoveryReplays + 1
          : state.recoveryReplays,
      duplicateEffects:
        mode === "prompt-only" && state.effectStatus === "uncertain" && state.uncertainOccurred
          ? state.duplicateEffects + 1
          : state.duplicateEffects,
    });
  }
  next.push({
    ...state,
    admissions: state.admissions + 1,
    knowledgeFacts: state.knowledgeFacts + 1,
    coordinationWrites: state.coordinationWrites + (mode === "enforced" ? 3 : 1),
  });
  if (mode === "prompt-only") {
    next.push({ ...state, knowledgeFacts: state.knowledgeFacts + 1 });
  }
  if (state.messageDeliveries === 0) {
    next.push({ ...state, messageDeliveries: 1 });
  }
  if (mode === "prompt-only" && state.messageDeliveries > 0) {
    next.push({ ...state, duplicateMessages: state.duplicateMessages + 1 });
  }
  return next;
}

function violates(state: ModelState): boolean {
  return (
    state.authorizedAttempts.length > 1 ||
    state.staleEffects > 0 ||
    state.duplicateEffects > 0 ||
    state.recoveryReplays > 0 ||
    state.knowledgeFacts > state.admissions ||
    state.duplicateMessages > 0
  );
}

export function checkSwarmModel(mode: VerificationMode, maxDepth = 7): ModelCheckReport {
  let frontier = [INITIAL];
  const visited = new Map([[key(INITIAL), INITIAL]]);
  for (let depth = 0; depth < maxDepth; depth += 1) {
    const following: ModelState[] = [];
    for (const state of frontier) {
      for (const candidate of transitions(state, mode)) {
        const identity = key(candidate);
        if (visited.has(identity)) continue;
        visited.set(identity, candidate);
        following.push(candidate);
      }
    }
    frontier = following;
  }
  const states = [...visited.values()];
  const violating = states.filter(violates);
  return {
    mode,
    statesExplored: states.length,
    safetyViolationRate: violating.length / states.length,
    duplicateEffects: Math.max(0, ...states.map((state) => state.duplicateEffects)),
    recoveryReplays: Math.max(0, ...states.map((state) => state.recoveryReplays)),
    knowledgePollution: Math.max(
      0,
      ...states.map((state) => state.knowledgeFacts - state.admissions),
    ),
    coordinationWrites: Math.max(0, ...states.map((state) => state.coordinationWrites)),
  };
}

export function runFaultBenchmark(maxDepth = 7): {
  readonly enforced: ModelCheckReport;
  readonly promptOnly: ModelCheckReport;
} {
  return {
    enforced: checkSwarmModel("enforced", maxDepth),
    promptOnly: checkSwarmModel("prompt-only", maxDepth),
  };
}
