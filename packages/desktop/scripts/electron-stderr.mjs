import { PassThrough, Transform } from "node:stream";

const MACOS_INPUT_METHOD_DIAGNOSTIC =
  /^(?:\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} )?.+\[\d+:\d+\] error messaging the mach port for IMKCFRunLoopWakeUpReliable\r?\n?$/;

export function createElectronStderrFilter(platform = process.platform) {
  if (platform !== "darwin") return new PassThrough();

  let pending = Buffer.alloc(0);
  return new Transform({
    transform(chunk, _encoding, callback) {
      const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      pending = pending.length === 0 ? bytes : Buffer.concat([pending, bytes]);

      for (let newline = pending.indexOf(0x0a); newline >= 0; newline = pending.indexOf(0x0a)) {
        const line = pending.subarray(0, newline + 1);
        pending = pending.subarray(newline + 1);
        if (!MACOS_INPUT_METHOD_DIAGNOSTIC.test(line.toString("utf8"))) this.push(line);
      }
      callback();
    },
    flush(callback) {
      if (pending.length > 0 && !MACOS_INPUT_METHOD_DIAGNOSTIC.test(pending.toString("utf8"))) {
        this.push(pending);
      }
      callback();
    },
  });
}

export function forwardElectronStderr(source, destination = process.stderr) {
  source?.pipe(createElectronStderrFilter()).pipe(destination, { end: false });
}
