# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting:
**Security → Report a vulnerability** on https://github.com/Eyshika/burnlens, with steps to
reproduce and the version (`burnlens --version`). Expect an acknowledgement within a week.

## Scope and threat model

Burnlens is a local tool.

- The dashboard binds to `127.0.0.1` by default. Do not expose it to a network: the OTLP
  receiver has no authentication.
- The action API requires a loopback server and a per-server token. It is a local workflow, not
  an access-control system.
- Transcripts can contain secrets pasted by you or printed by tools. Burnlens reads them in place
  and does not copy them anywhere. The optional `explain` feature sends a digest (prompts, file
  names, command heads, sizes) to the endpoint you configure, never file contents or tool output.
- API keys are read from environment variables only and are never written to disk.
- `burnlens capture` runs the argv you give it without a shell and keeps full logs under
  `~/.burnlens/captures/`. Those logs may contain whatever your command printed.

Reports about path traversal in the static server, token handling in the action API, hook
output injection, or unintended data egress are especially welcome.
