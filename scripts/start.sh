#!/bin/bash
# One command for the pet (PLAN.md G2). Checks first (scripts/preflight.py), then:
#
#   scripts/start.sh [run] [petd args]   in this terminal; Ctrl-C stops it (the default)
#   scripts/start.sh start [petd args]    in the background, as the systemd user unit "petd"
#   scripts/start.sh stop | restart | status
#   scripts/start.sh log                  follow the current run's petd.log
#   scripts/start.sh check                the checks only
#   scripts/start.sh install | uninstall  start petd whenever WSL starts (README.md: Starting with WSL)
#
# In the background, systemd stops petd with SIGINT (a clean shutdown, as Ctrl-C)
# and restarts it if it crashes. Without an installed unit, "start" runs a
# transient one; with one, it starts that.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
UNIT=petd
UNIT_FILE="$HOME/.config/systemd/user/$UNIT.service"

cd "$ROOT"
[ -x "$PY" ] || { echo "no venv at $ROOT/.venv: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
mkdir -p runtime

running() { systemctl --user is-active --quiet "$UNIT" 2>/dev/null; }
need_systemd() {
    systemctl --user show-environment >/dev/null 2>&1 \
        || { echo "no systemd user session here: use '$0 run'"; exit 1; }
}

unit_text() {
    cat <<EOF
[Unit]
Description=robot-pet: petd
After=network-online.target

[Service]
WorkingDirectory=$ROOT
ExecStartPre=$PY $ROOT/scripts/preflight.py
ExecStart=$PY -m petd
KillSignal=SIGINT
TimeoutStopSec=30
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF
}

cmd="${1:-run}"
case "$cmd" in
    run)
        shift || true
        if running; then echo "petd is already running in the background ('$0 stop' first)"; exit 1; fi
        "$PY" scripts/preflight.py
        exec "$PY" -m petd "$@"
        ;;
    start)
        shift || true
        need_systemd
        if running; then echo "petd is already running"; exit 0; fi
        "$PY" scripts/preflight.py
        if [ -f "$UNIT_FILE" ]; then
            [ $# -eq 0 ] || echo "the installed unit runs plain petd: ignoring $*"
            systemctl --user start "$UNIT"
        else
            systemd-run --user --unit="$UNIT" --working-directory="$ROOT" \
                --property=KillSignal=SIGINT --property=TimeoutStopSec=30 \
                --property=Restart=on-failure --property=RestartSec=10 \
                "$PY" -m petd "$@" >/dev/null
        fi
        sleep 2
        systemctl --user --no-pager --lines=0 status "$UNIT" | head -3
        echo "logs: $0 log   (or runtime/logs/latest/petd.log)"
        ;;
    stop)
        need_systemd
        if running; then systemctl --user stop "$UNIT"; echo "petd stopped"; else echo "petd is not running"; fi
        ;;
    restart)
        "$0" stop
        "$0" start
        ;;
    status)
        need_systemd
        if running; then
            systemctl --user --no-pager --lines=0 status "$UNIT" | head -4
        else
            echo "petd is not running in the background"
        fi
        [ -e runtime/logs/latest ] && echo "latest run: runtime/logs/$(readlink runtime/logs/latest)"
        ;;
    log)
        exec tail -F runtime/logs/latest/petd.log
        ;;
    check)
        exec "$PY" scripts/preflight.py
        ;;
    install)
        need_systemd
        mkdir -p "$(dirname "$UNIT_FILE")"
        unit_text > "$UNIT_FILE"
        systemctl --user daemon-reload
        systemctl --user enable "$UNIT"
        echo "installed $UNIT_FILE: petd starts with the user's systemd session."
        echo "To start it without anyone logging in: loginctl enable-linger $USER"
        echo "For WSL to start at all after a Windows reboot, see README.md (Starting with WSL)."
        ;;
    uninstall)
        need_systemd
        if [ -f "$UNIT_FILE" ]; then
            systemctl --user disable "$UNIT" || true
            mv "$UNIT_FILE" "$UNIT_FILE.removed"
            systemctl --user daemon-reload
            echo "disabled; the unit file is kept as $UNIT_FILE.removed"
        else
            echo "not installed"
        fi
        ;;
    -h|--help|help)
        sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        # petd's own flags (--fake, --echo, ...) go to "run"
        if [[ "$cmd" == -* ]]; then exec "$0" run "$@"; fi
        echo "unknown command: $cmd ('$0 help')"; exit 2
        ;;
esac
