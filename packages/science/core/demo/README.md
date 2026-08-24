# dsh-science demo

`runScienceDemo(ctx.science, sessionId, signal?)` runs one local-only tour through the public
Science service:

1. create a project and Notebook;
2. execute a Python cell that creates a dataset and figure artifact;
3. define an Experiment, start and finish a reproducible Run;
4. register and semantically edit a matplotlib Figure;
5. create a Typst document and accept a source-selection revision;
6. record a question, hypothesis, claim, and supporting evidence;
7. read the RO-Crate 1.3 Research Object and create a content-addressed project export.

The example is executable in the repository through `packages/science/core/tests/demo.test.ts`.
Repository tests select the explicit isolated runtime so they require no external kernel; the
installed product defaults to the configured local JupyMCP controller. Neither mode returns a host
path or performs an implicit dependency download or remote Jupyter connection.
