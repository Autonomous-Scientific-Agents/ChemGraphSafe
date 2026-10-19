# Local verification

Verified on macOS with Python 3.12.11 on 2026-10-19.

- Locked installation: `uv sync --locked --extra dev` passed (32 resolved packages).
- Ruff lint and formatting checks passed.
- Pytest: 35 passed, using harmless fixtures and mock HTTP; no paid inference.
- Offline CLI `evaluate` then `report` passed: 10 cases, 9 decisions and 1 native abstention;
  the report clearly identifies MOCK/OFFLINE results and includes both diagnostic baselines.
- Real SciHazard import passed both pinned file checksums. 250 chemistry source records;
  22 native/split conflicts quarantined; 228 eligible groups; 100 selected groups, 50/50 classes.
- Real data freeze validated: development 10 benign + 10 harmful; evaluation 40 + 40;
  zero policy annotations, as requested for the initial source-only workflow.
- OpenRouter key was absent, so live Jev smoke/evaluation were not run. The wire contract,
  malformed response handling, retries, budgets and native refusals were tested with mock HTTP.
- Live LLM work, training, and calibration remain contributor tasks. CI is configured but has
  not run on a remote host; no commit or remote publication was performed.

Pilot examples SHA256: `016a49afa9f3f9495c240309ba2d799f83f9e43eddfc6c882181173e145afb94`.
Split/order hash: `7fa9c77d7ca942937e31248aa1a387f6150b6b519ee2e9b482d21327c85e3ded`.
The local freeze manifest stores these plus configuration, preparation and code provenance.

The local sandbox could not resolve external hosts and caused uv's macOS system-configuration
lookup to fail. Public source download and locked sync succeeded with approved sandbox escalation.
No raw chemistry requests were printed in verification output or committed to the project.

See `PLAN-REVIEW-LOG.md` for the independent Claude inspection transcript and dispositions.
