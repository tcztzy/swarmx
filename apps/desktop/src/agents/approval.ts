import { z } from "zod";
import type { Interaction, Observer } from "./types.js";

/** Validate the user's choice without widening or rewriting the native offered decision. */
export async function requestApproval<T>(
  observer: Observer,
  id: string,
  title: string,
  toolId: string,
  choices: readonly (Omit<NonNullable<Interaction["approval"]>["choices"][number], "answer"> & {
    answer: T;
  })[],
  signal?: AbortSignal,
): Promise<T | undefined> {
  const answer = await observer.interact(
    {
      id,
      title,
      schema: {
        type: "object",
        properties: { optionId: { type: "string", enum: choices.map((choice) => choice.id) } },
        required: ["optionId"],
        additionalProperties: false,
      },
      approval: {
        toolId,
        choices: choices.map(({ answer: _answer, ...choice }) => ({
          ...choice,
          answer: { optionId: choice.id },
        })),
      },
    },
    signal,
  );
  if (answer === undefined || signal?.aborted) return undefined;
  const { optionId } = z.strictObject({ optionId: z.string() }).parse(answer);
  const selected = choices.find((choice) => choice.id === optionId);
  if (!selected) throw new Error("Unknown native approval choice.");
  return selected.answer;
}
