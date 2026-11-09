#!/bin/bash
# Comandi lunghi staccati dalla sessione di Claude CLI: continuano anche se la
# sessione finisce, e una ripresa li può aspettare di nuovo.
#
#   bg.sh start NOME COMANDO ARGOMENTI...   avvia; output in scratch/logs/NOME.log
#   bg.sh wait NOME                         aspetta fino a 9 minuti, poi stampa
#                                           "finished exit=N" oppure "running",
#                                           e le ultime righe del log
#
# Per redirezioni o pipe usa: bg.sh start NOME /bin/bash -c '...'

set -u
usage() { echo "uso: bg.sh start NOME COMANDO ARGOMENTI... | bg.sh wait NOME" >&2; exit 2; }

cd "$(git rev-parse --show-toplevel)" || exit 2
action=${1:-}
name=${2:-}
case $name in ''|*/*|.*) usage ;; esac
dir=scratch/logs
mkdir -p "$dir"
log=$dir/$name.log
pidf=$dir/$name.pid
exitf=$dir/$name.exit

running() { [ -f "$pidf" ] && ps -p "$(cat "$pidf")" >/dev/null 2>&1; }

case $action in
  start)
    shift 2
    [ $# -gt 0 ] || usage
    if running; then echo "already running: $name (usa: bg.sh wait $name)"; exit 1; fi
    rm -f "$exitf"
    # setsid: il comando esce dal gruppo di processi di Claude CLI.
    nohup python3 -I -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
      /bin/bash -c 'log=$1; exitf=$2; shift 2; "$@" > "$log" 2>&1; echo $? > "$exitf.tmp"; mv "$exitf.tmp" "$exitf"' \
      bg "$log" "$exitf" "$@" </dev/null >/dev/null 2>&1 &
    echo $! > "$pidf"
    sleep 1   # lascia al comando il tempo di staccarsi prima di tornare
    echo "started $name (pid $(cat "$pidf")), log $log"
    ;;
  wait)
    if [ ! -f "$pidf" ] && [ ! -f "$exitf" ]; then echo "not started: $name"; exit 1; fi
    i=0
    while [ ! -f "$exitf" ] && running && [ "$i" -lt 54 ]; do sleep 10; i=$((i + 1)); done
    if [ -f "$exitf" ]; then
      echo "finished exit=$(cat "$exitf")"
    elif running; then
      echo "running (ripeti: bg.sh wait $name)"
    else
      echo "finished exit=unknown (il processo è terminato senza codice di uscita)"
    fi
    if [ -f "$log" ]; then tail -n 15 "$log"; fi
    ;;
  *) usage ;;
esac
