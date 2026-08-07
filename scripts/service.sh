#!/usr/bin/env bash
# Manage both services as systemd units.
#
#   scripts/service.sh install [realtime|batch]
#   scripts/service.sh {start|stop|restart|status|logs|uninstall} [realtime|batch]
#
# Units are generated from deploy/*.template with this checkout's real paths, so
# the project works from any directory and under any user without editing files
# by hand.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYSTEMD_DIR=/etc/systemd/system
SERVICES="realtime batch"

ACTION="${1:-}"
TARGET="${2:-all}"

if [ "${TARGET}" != "all" ]; then
  case "${TARGET}" in
    realtime|batch) SERVICES="${TARGET}" ;;
    *) echo "unknown service: ${TARGET} (expected realtime or batch)" >&2; exit 1 ;;
  esac
fi

unit_name() { echo "breeze-$1.service"; }

do_install() {
  local name template dest
  local user="${SUDO_USER:-$(id -un)}"
  local group; group="$(id -gn "${user}")"
  local python; python="$(command -v python3)"

  for svc in ${SERVICES}; do
    name="$(unit_name "${svc}")"
    template="${REPO_ROOT}/deploy/breeze-${svc}.service.template"
    dest="${SYSTEMD_DIR}/${name}"
    echo "==> Installing ${name}"
    sed -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
        -e "s|@USER@|${user}|g" \
        -e "s|@GROUP@|${group}|g" \
        -e "s|@PYTHON@|${python}|g" \
        "${template}" | sudo tee "${dest}" >/dev/null
  done

  sudo systemctl daemon-reload
  for svc in ${SERVICES}; do
    sudo systemctl enable --now "$(unit_name "${svc}")"
  done
  echo "==> Installed and started. Check with: $0 status"
}

do_uninstall() {
  for svc in ${SERVICES}; do
    name="$(unit_name "${svc}")"
    echo "==> Removing ${name}"
    sudo systemctl disable --now "${name}" 2>/dev/null || true
    sudo rm -f "${SYSTEMD_DIR}/${name}"
  done
  sudo systemctl daemon-reload
}

case "${ACTION}" in
  install)   do_install ;;
  uninstall) do_uninstall ;;
  start|stop|restart)
    for svc in ${SERVICES}; do sudo systemctl "${ACTION}" "$(unit_name "${svc}")"; done
    ;;
  status)
    for svc in ${SERVICES}; do
      systemctl status --no-pager --lines=0 "$(unit_name "${svc}")" || true
    done
    ;;
  logs)
    if [ "${TARGET}" = "all" ]; then
      journalctl -f -u "$(unit_name realtime)" -u "$(unit_name batch)"
    else
      journalctl -f -u "$(unit_name "${TARGET}")"
    fi
    ;;
  *)
    sed -n '2,10p' "$0"
    exit 1
    ;;
esac
