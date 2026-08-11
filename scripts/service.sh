#!/usr/bin/env bash
# Manage both services as systemd units.
#
#   scripts/service.sh install [realtime|batch] [--user]
#   scripts/service.sh {start|stop|restart|status|logs|uninstall} [realtime|batch] [--user]
#
# Units are generated from deploy/*.template with this checkout's real paths, so
# the project works from any directory and under any user without editing files
# by hand.
#
# --user installs into ~/.config/systemd/user instead of /etc/systemd/system,
# for a box where you have no root. Such a unit only survives logout if the
# account lingers; install checks and tells you the command if it does not.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICES="realtime batch"

ACTION=""
TARGET="all"
USER_MODE=0

for arg in "$@"; do
  case "${arg}" in
    --user)         USER_MODE=1 ;;
    realtime|batch) TARGET="${arg}" ;;
    all)            TARGET="all" ;;
    -*)             echo "unknown option: ${arg}" >&2; exit 1 ;;
    *)
      if [ -z "${ACTION}" ]; then
        ACTION="${arg}"
      else
        echo "unknown service: ${arg} (expected realtime or batch)" >&2
        exit 1
      fi
      ;;
  esac
done

if [ "${TARGET}" != "all" ]; then
  SERVICES="${TARGET}"
fi

if [ "${USER_MODE}" -eq 1 ]; then
  SYSTEMD_DIR="${HOME}/.config/systemd/user"
  SUDO=""
  SYSTEMCTL="systemctl --user"
  JOURNALCTL="journalctl --user"
  USER_FLAG=" --user"
else
  SYSTEMD_DIR=/etc/systemd/system
  SUDO="sudo"
  SYSTEMCTL="sudo systemctl"
  JOURNALCTL="journalctl"
  USER_FLAG=""
fi

unit_name() { echo "breeze-$1.service"; }

# A checkout with its own virtualenv should run from it. Falls back to whatever
# python3 is on PATH, which is what a bare Jetson has.
pick_python() {
  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    echo "${REPO_ROOT}/.venv/bin/python"
  else
    command -v python3
  fi
}

# Without lingering, a user unit dies at logout and never starts at boot.
warn_if_not_lingering() {
  local who="${USER:-$(id -un)}"
  if [ "$(loginctl show-user "${who}" --property=Linger --value 2>/dev/null)" != "yes" ]; then
    echo "==> NOTE: ${who} does not linger, so these units stop at logout and"
    echo "    will not start at boot. Enable it with:"
    echo "      sudo loginctl enable-linger ${who}"
  fi
}

do_install() {
  local name template dest python
  local user="${SUDO_USER:-$(id -un)}"
  local group; group="$(id -gn "${user}")"
  python="$(pick_python)"

  [ "${USER_MODE}" -eq 1 ] && mkdir -p "${SYSTEMD_DIR}"

  for svc in ${SERVICES}; do
    name="$(unit_name "${svc}")"
    template="${REPO_ROOT}/deploy/breeze-${svc}.service.template"
    dest="${SYSTEMD_DIR}/${name}"
    echo "==> Installing ${name}"
    # A user unit runs as its owner by definition, and multi-user.target is not
    # a thing in the user manager, so both have to go.
    if [ "${USER_MODE}" -eq 1 ]; then
      sed -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
          -e "s|@PYTHON@|${python}|g" \
          -e "/^User=/d" \
          -e "/^Group=/d" \
          -e "s|^WantedBy=multi-user.target$|WantedBy=default.target|" \
          "${template}" > "${dest}"
    else
      sed -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
          -e "s|@USER@|${user}|g" \
          -e "s|@GROUP@|${group}|g" \
          -e "s|@PYTHON@|${python}|g" \
          "${template}" | ${SUDO} tee "${dest}" >/dev/null
    fi
  done

  ${SYSTEMCTL} daemon-reload
  for svc in ${SERVICES}; do
    ${SYSTEMCTL} enable --now "$(unit_name "${svc}")"
  done
  [ "${USER_MODE}" -eq 1 ] && warn_if_not_lingering
  echo "==> Installed and started. Check with: $0 status${USER_FLAG}"
}

do_uninstall() {
  for svc in ${SERVICES}; do
    name="$(unit_name "${svc}")"
    echo "==> Removing ${name}"
    ${SYSTEMCTL} disable --now "${name}" 2>/dev/null || true
    ${SUDO} rm -f "${SYSTEMD_DIR}/${name}"
  done
  ${SYSTEMCTL} daemon-reload
}

case "${ACTION}" in
  install)   do_install ;;
  uninstall) do_uninstall ;;
  start|stop|restart)
    for svc in ${SERVICES}; do ${SYSTEMCTL} "${ACTION}" "$(unit_name "${svc}")"; done
    ;;
  status)
    for svc in ${SERVICES}; do
      ${SYSTEMCTL} status --no-pager --lines=0 "$(unit_name "${svc}")" || true
    done
    ;;
  logs)
    if [ "${TARGET}" = "all" ]; then
      ${JOURNALCTL} -f -u "$(unit_name realtime)" -u "$(unit_name batch)"
    else
      ${JOURNALCTL} -f -u "$(unit_name "${TARGET}")"
    fi
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
