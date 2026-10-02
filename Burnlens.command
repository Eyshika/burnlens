#!/bin/zsh
# Double-click to open the dashboard. Uses ./.venv if present, else the conda env, else python3.
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m burnlens ui
fi
if command -v conda >/dev/null 2>&1 && conda env list | grep -q '^burnlens '; then
  eval "$(conda shell.zsh hook)"
  conda activate burnlens
fi
exec python3 -m burnlens ui
