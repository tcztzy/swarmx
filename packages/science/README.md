# Science packages

Host-side scientific capabilities live in this group. The first vertical slice is one package,
[`core/`](core), because its client-safe contracts, append-only journal, projections, and
`ctx.science` service share one release and ownership boundary.

Notebook execution, writing revisions, visualization editing, operational research facts,
Experiment/Run records, artifacts, RO-Crate exports, and agent tools stay inside this package as focused modules until a
real independent consumer or release cadence justifies another package.

`core/config/agent-presets/dsh-science` publishes the read-only **Science mode / 科学模式** roster entry. Its agent
composition is the complete locked DSH `standard` preset followed by the Science tool and model-contract
plugins; the Host `ctx.science` service and browser artifact UI remain globally composed.
