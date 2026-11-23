#!/usr/bin/env bash
# In the interactive Bash shell that started training:
#   source scripts/run_after_training.sh %1 /path/to/next_training.sh

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    exec python3 "$(dirname "$0")/watch_dspark_training.py" "$@"
fi

if [[ $# -lt 2 || $1 != %* ]]; then
    echo "Usage: source run_after_training.sh %JOB /path/to/next_training.sh [args...]" >&2
    return 2
fi
if [[ ! -f $2 || ! -r $2 ]]; then
    printf 'Cannot read next training script: %s\n' "$2" >&2
    return 2
fi

# fg resumes a suspended job, waits for it, and returns its exit status.
if fg "$1"; then
    printf 'Training completed. Starting %s\n' "$2"
    bash -- "$2" "${@:3}"
else
    printf 'Job %s returned status %s. Skipping %s\n' "$1" "$?" "$2" >&2
    return 1
fi
