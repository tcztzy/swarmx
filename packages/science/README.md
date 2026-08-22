# Science packages

Host-side scientific capabilities live in this group. The first vertical slice is one package,
[`core/`](core), because its client-safe contracts, append-only journal, projections, and
`ctx.science` service share one release and ownership boundary.

Notebook execution, writing revisions, visualization editing, Research Map facts, Experiment/Run
records, artifacts, exports, and agent tools stay inside this package as focused modules until a
real independent consumer or release cadence justifies another package.
