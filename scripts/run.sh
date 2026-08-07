#!/usr/bin/env bash
# Run a service in the foreground, for development and for checking a fresh
# install before committing to systemd.
#
#   scripts/run.sh realtime
#   scripts/run.sh batch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

case "${1:-}" in
  realtime) exec python3 services/realtime/server.py ;;
  batch)    exec python3 services/batch/app.py ;;
  *)
    echo "usage: $0 {realtime|batch}" >&2
    exit 1
    ;;
esac
