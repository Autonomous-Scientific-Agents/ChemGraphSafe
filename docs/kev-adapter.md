# Local Kev adapter

The `kev` adapter calls a separately managed Kev System One server on literal loopback HTTP.
It sends only request text, context, and the classification policy.
It does not install a runtime, download weights, or start a server.
The harness needs no additional Python dependencies.

## Server setup

Use the [upstream Kev source](https://github.com/jaredpalmer/kev), not an unrelated package with the same name.
The example configuration pins these identities:

| Item | Pin |
|---|---|
| Serving source | `1f04cfa28e6a708485b28a9585010b9e436b776e` |
| Kev checkpoint | `jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f` |
| Base checkpoint | `Qwen/Qwen3.5-0.8B-Base` |
| Device / dtype | CPU / float32 |
| Head temperature | `2.3510958125672174` |

The temperature comes from the checkpoint head metadata. The rounded model-card value is insufficient for the identity check.
A separate prototype used Python 3.12, Torch 2.8.0+cpu, Transformers 5.20.2, and PEFT 0.21.2.
These are deployment observations, not new harness dependencies or a complete server lockfile.

For a CPU environment, use `uv` and keep the serving dependencies separate:

```sh
mkdir -p .local/servers
git clone --filter=blob:none https://github.com/jaredpalmer/kev.git .local/servers/kev
git -C .local/servers/kev checkout --detach 1f04cfa28e6a708485b28a9585010b9e436b776e
uv venv .local/kev-env --python 3.12
uv pip install --python .local/kev-env/bin/python torch==2.8.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .local/kev-env/bin/python \
  -e '.local/servers/kev[serve]' transformers==5.20.2 peft==0.21.2
KEV_BACKEND=torch KEV_DTYPE=fp32 KEV_TRUNCATE_STATES=0 KEV_DATE_FACTS=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.local/kev-env/bin/python -m kev.serve \
  --run jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f \
  --device cpu --host 127.0.0.1 --port 8008
```

Server installation and startup can download packages and model artifacts.
Keep inherited `KEV_*` overrides consistent with the upstream defaults.
Use the exact source checkout and retain its deployment lock and artifact checksums separately.
The adapter verifies only the metadata that the stock server exposes.
It cannot attest the serving source revision, base checkpoint revision, or every environment override.
The configured runtime revision is an operator declaration recorded in run identity.

## Harness configuration and execution

Use `configs/kev-local.toml`.
The endpoint must use `127.0.0.1` or `::1` with `/v1/systemone`.
Hostname aliases, non-loopback addresses, embedded credentials, query parameters, and fragments are rejected.
Other HTTP providers retain their HTTPS and credential requirements.
Environment proxy settings are disabled for the local client.

Authentication is optional. If the server requires a key, set `api_key_env` and export that variable.
A configured but missing credential causes failure before inference.

These commands perform real inference and consume shared attempts:

```sh
uv run chemgraphsafe smoke --config configs/kev-local.toml
uv run chemgraphsafe data prepare --config configs/kev-local.toml
uv run chemgraphsafe data freeze --config configs/kev-local.toml
uv run chemgraphsafe evaluate --config configs/kev-local.toml
uv run chemgraphsafe report --run runs/kev-local-v1
```

The shared ledger remains `.local/state/pilot-budget.sqlite`, with its existing limits.
`budget_mode="attempts_only"` is required. Local monetary cost remains unavailable.
Use new data and run directories when identities change. Never replace an existing freeze.
Kev cannot serve as a curator because it does not provide the required curation reason codes.

## Parsing, identity, and limits

Construction checks `/v1/models` without inference or attempt reservation.
Before each attempt, the adapter checks checkpoint, base model name, device, dtype, backend,
head temperature, and disabled state truncation.
Identity drift stops further requests through the runner's fatal-error handling.
The model field in a prediction records the checkpoint verified by preflight, not the response's echoed alias.

Successful decisions require a complete numeric distribution matching the active policy.
The default policy has ALLOW/REVIEW/BLOCK labels. The opt-in binary policy has ALLOW/BLOCK labels.
Changing policy requires a new data version and freeze; an existing three-way run cannot become binary.
Malformed answers and explicit truncation become errors. An optional boolean abstention remains separate from REVIEW.
Client latency and attempts use the shared HTTP transport.
Valid native input-token counts are retained.
Upstream output counters do not establish comparable generated-token usage, so output tokens remain unavailable.
Cost-like fields from local responses are not treated as dollar charges.

Native distributions do not establish chemistry calibration or useful triage performance.
The offline tests use owned harmless requests and mock HTTP. They do not launch models.

```sh
uv run pytest -q tests/test_kev.py
```
