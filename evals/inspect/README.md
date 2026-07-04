# SwarmX Inspect Evals

This directory contains a thin Inspect adapter around the SwarmX eval JSON
contract.

## Environment Setup

Install `uv` if it is not already available:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, create `.venv` and install the Inspect dependency
group from `pyproject.toml`:

```sh
uv sync --group inspect
```

Build the TypeScript packages before running Inspect from source. The default
adapter command is `node packages/cli/dist/cli.js`, so it expects this build
output:

```sh
pnpm -r build
```

## Run The Smoke Eval

The default smoke eval includes two samples:

- `fixtures/echo.swarm.json`: a deterministic positive sample using
  `backend: { "type": "echo" }`
- `fixtures/missing-root.swarm.json`: an intentionally invalid runtime sample
  that verifies `swarmx eval-run` returns a parseable JSON result with an
  `error` field and stable metrics

Neither sample requires model credentials.

```sh
uv run --group inspect inspect eval evals/inspect/tasks.py@swarmx_smoke --model none
```

The repository also exposes convenience scripts:

```sh
pnpm test:inspect
pnpm run ci
```

`pnpm test:inspect` runs the Python adapter unit tests and the Inspect smoke
eval. `pnpm run ci` runs Vitest, TypeScript/Electron builds, `uv sync --group
inspect`, and the Inspect checks.

Open Inspect logs with:

```sh
uv run --group inspect inspect view
```

Inspect tasks are built from datasets, solvers, and scorers. The adapter maps
those to:

- dataset: JSONL rows in `evals/inspect/datasets/`
- solver: `swarmx_solver`, which calls `swarmx eval-run`
- scorer: `swarmx_contract_scorer`, which checks output, error, trace, and metrics

## Dataset Fields

Each JSONL row supports:

- `id`: sample id
- `input`: user prompt
- `target`: required output substring, optional
- `arguments`: full `swarmx eval-run --input-json` object; overrides `input`
- `config`: per-sample Swarm config path; overrides the task default config
- `expect_error_contains`: expected error substring
- `expect_output_contains`: expected output substring
- `expected_nodes`: exact trace node sequence
- `expected_steps`: exact `metrics.steps`
- `max_steps`: upper bound for `metrics.steps`
- `expected_tool_calls`: exact `metrics.toolCalls`

For a real eval, pass a Swarm config:

```sh
uv run --group inspect inspect eval evals/inspect/tasks.py@swarmx_smoke \
  --model none \
  -T config=/absolute/path/to/swarm.json \
  -T dataset=/absolute/path/to/cases.jsonl
```

If you have installed the `swarmx` binary elsewhere, override the command:

```sh
uv run --group inspect inspect eval evals/inspect/tasks.py@swarmx_smoke \
  --model none \
  -T command=swarmx
```

The Inspect API concepts used here are documented in the official pages for
[custom solvers](https://inspect.aisi.org.uk/solvers.html),
[JSON datasets](https://inspect.aisi.org.uk/datasets.html), and
[custom scorers](https://inspect.aisi.org.uk/tutorial.html#custom-scorers).
