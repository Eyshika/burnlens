# Contributing to Burnlens

Thanks for helping. Burnlens is a small, dependency-free Python package; keeping it that way is
a goal.

## Set up

```bash
git clone https://github.com/Eyshika/burnlens.git
cd burnlens
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check burnlens tests
pytest -q
burnlens ui --demo        # try your change on synthetic data
```

## Ground rules

- **No runtime dependencies.** The standard library only. Dev tools go in the `dev` extra.
- **No secrets, no real data.** Never commit keys, transcripts, exports or `.env` files. Fixtures
  are synthetic; `burnlens/demo.py` generates the sample data.
- **Measured vs inferred.** Anything the UI or CLI states as a saving must be measured. A
  projection or an attribution is labelled as one (`savings_status`), and quality is never
  implied to be equal when a cheaper model is suggested.
- **Local by default.** A feature that sends data off the machine must be opt-in, documented in
  the README privacy section, and send a digest, never file contents.
- **Config over code.** Thresholds belong in `burnlens.example.toml`, prices in
  `burnlens/prices.toml`. Update the `as_of` date only when you actually re-checked the prices.
- **Tests go next to the area they cover.** Add a test that fails without your change, and for a
  new feature drive it through the real entry point so an unwired function cannot pass.
- **Comments are short.** One or two lines, only where the code would otherwise mislead.

## Adding an agent adapter

Native readers live in `burnlens/adapter_*.py` and register in `burnlens/adapters.py`. If the
agent exposes no local history, document the export route in `docs/INGEST.md` instead. Include a
small synthetic fixture and say which fields are exact and which are approximated.

## Pull requests

1. Fork, branch from `main`, keep the change focused.
2. Run `ruff check burnlens tests` and `pytest -q`.
3. Fill in the PR template, including what you tried on `burnlens ui --demo`.
4. Use conventional prefixes in commit subjects: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

## Reporting bugs and ideas

Use the issue templates. For security problems, see [SECURITY.md](SECURITY.md) and do not open a
public issue.
