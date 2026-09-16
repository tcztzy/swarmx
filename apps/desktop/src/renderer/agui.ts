import { AbstractAgent, type BaseEvent, type RunAgentInput } from "@ag-ui/client";
import { EventType } from "@ag-ui/core";
import { Observable } from "rxjs";
import { AgUiEventMessageSchema } from "../bridge-contract.js";
import { bridge } from "./bridge.js";

/** AG-UI client over the Electron IPC stream: one run at a time per thread. */
export class IpcAgent extends AbstractAgent {
  private active: (() => void) | undefined;
  constructor(
    private readonly harness: string,
    threadId: string,
  ) {
    super({ agentId: harness, threadId });
  }
  override run(input: RunAgentInput): Observable<BaseEvent> {
    return new Observable<BaseEvent>((subscriber) => {
      let finished = false;
      const finish = () => {
        if (finished) return;
        finished = true;
        subscriber.complete();
      };
      const unsubscribe = bridge().agui.subscribe((raw) => {
        const message = AgUiEventMessageSchema.parse(raw);
        if (message.threadId !== input.threadId || finished) return;
        if (message.event !== undefined) {
          const event = message.event as BaseEvent;
          subscriber.next(event);
          if (event.type === EventType.RUN_FINISHED || event.type === EventType.RUN_ERROR) finish();
        }
        if (message.error !== undefined && !finished) {
          finished = true;
          subscriber.error(new Error(message.error));
        }
        if (message.done === true) finish();
      });
      this.active = finish;
      void bridge()
        .agui.start({ agent: this.harness, input })
        .catch((error: unknown) => {
          if (finished) return;
          finished = true;
          subscriber.error(error instanceof Error ? error : new Error(String(error)));
        });
      return () => {
        unsubscribe();
        this.active = undefined;
      };
    });
  }
  override abortRun(): void {
    this.active?.();
    void bridge()
      .agui.cancel({ agent: this.harness, threadId: this.threadId })
      .catch(() => undefined);
  }
  override clone(): IpcAgent {
    const copy = super.clone() as IpcAgent;
    copy.active = undefined;
    return copy;
  }
}
