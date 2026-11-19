#!/usr/bin/env bash
set -euo pipefail

if (( $# > 4 )); then
    echo "Usage: bash benchmark.sh [config [speculation [RPM [run_id]]]]" >&2
    exit 2
fi

exec bash "$(dirname "$0")/profile.sh" benchmark "${3:-120}" "${1:-baseline}" "${2:-off}" "${4:-}"
