#!/usr/bin/env bash
set -euo pipefail

if (( $# > 4 )); then
    echo "Usage: bash load_test.sh [RPM [config [speculation [run_id]]]]" >&2
    exit 2
fi

exec bash "$(dirname "$0")/profile.sh" load "${1:-120}" "${2:-baseline}" "${3:-off}" "${4:-}"
