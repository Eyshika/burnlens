# Changelog

## 0.1.0 - 2026-10-02

First open-source release.

- Renamed the package and CLI from `tokprof` to `burnlens`; `tokprof` remains a CLI alias. State
  moves from `~/.tokprof` to `~/.burnlens`: run `mv ~/.tokprof ~/.burnlens` to keep history, then
  `burnlens install-hooks` to re-register hooks (the old hook entries are still recognised by
  `--remove`).
- `burnlens ui --demo` opens the dashboard on synthetic sessions.
- The 3D token flow map is bundled; the UI makes no third-party requests.
- Prices for Claude Opus 5.5 and Sonnet 5.5 added to `prices.toml`.
- The workflow card shows the first two proposals and collapses the rest.
- Added LICENSE (MIT), CONTRIBUTING, SECURITY, issue and PR templates, `.env.example`.
