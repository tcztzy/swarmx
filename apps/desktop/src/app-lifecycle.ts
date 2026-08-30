export interface BeforeQuitEvent {
  preventDefault(): void;
}

export interface SingleInstanceApplication {
  requestSingleInstanceLock(): boolean;
  quit(): void;
  on(event: "second-instance", listener: () => void): unknown;
}

export interface PrimaryWindow {
  focus(): void;
  isMinimized(): boolean;
  restore(): void;
  show(): void;
}

/** Acquire the sole product-state owner and route later launches to its visible window. */
export function acquirePrimaryInstance(
  application: SingleInstanceApplication,
  windows: () => readonly PrimaryWindow[],
): boolean {
  if (!application.requestSingleInstanceLock()) {
    application.quit();
    return false;
  }
  application.on("second-instance", () => {
    const window = windows()[0];
    if (window === undefined) return;
    if (window.isMinimized()) window.restore();
    window.show();
    window.focus();
  });
  return true;
}

export interface AppQuitCoordinatorOptions {
  dispose(): Promise<void>;
  quit(): void;
  reportFailure(error: unknown): void;
}

export interface OwnedAsyncResource {
  dispose(): Promise<void>;
}

/** Preserve the triggering failure while ensuring an already-owned resource is released. */
export async function disposeAfterFailure(
  resource: OwnedAsyncResource,
  failure: unknown,
): Promise<never> {
  try {
    await resource.dispose();
  } catch (cleanupError) {
    throw new AggregateError(
      [failure, cleanupError],
      "Desktop surface failure and platform cleanup both failed.",
      { cause: failure },
    );
  }
  throw failure;
}

/** Complete initial surface activation or release the platform before startup fails. */
export async function activateOwnedResource<Resource extends OwnedAsyncResource>(
  starting: Promise<Resource>,
  activate: (resource: Resource) => unknown,
): Promise<Resource> {
  const resource = await starting;
  try {
    activate(resource);
    return resource;
  } catch (error) {
    return disposeAfterFailure(resource, error);
  }
}

/** Collapse startup and shutdown observations of one fatal lifecycle failure. */
export function onceFailureReporter(report: (error: unknown) => void): (error: unknown) => void {
  let reported = false;
  return (error) => {
    if (reported) return;
    reported = true;
    report(error);
  };
}

/** Dispose the active platform, waiting for the exact in-flight boot when assignment has not landed. */
export async function disposeOwnedResource(
  current: () => OwnedAsyncResource | undefined,
  starting: () => Promise<OwnedAsyncResource> | undefined,
): Promise<void> {
  const resource = current() ?? (await starting());
  await resource?.dispose();
}

/** Hold application exit behind one owned cleanup attempt. */
export function createBeforeQuitHandler(
  options: AppQuitCoordinatorOptions,
): (event: BeforeQuitEvent) => void {
  let readyToQuit = false;
  let shutdown: Promise<void> | undefined;

  return (event) => {
    if (readyToQuit) return;
    event.preventDefault();
    if (shutdown !== undefined) return;

    try {
      shutdown = options.dispose();
    } catch (error) {
      options.reportFailure(error);
      return;
    }

    void shutdown.then(
      () => {
        readyToQuit = true;
        options.quit();
      },
      (error: unknown) => options.reportFailure(error),
    );
  };
}
