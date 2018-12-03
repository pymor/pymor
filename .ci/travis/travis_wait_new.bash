#!/usr/bin/env bash
set -e
set -u

TIMEOUT=$1

if [[ ${TIMEOUT} =~ ^[0-9]+$ ]]; then
    # looks like an integer, so we assume it's a timeout
    shift
else
    # default value
    TIMEOUT=20
fi

cmd="$@"

# running cmd at background
${cmd} &
cmd_pid=$!

# ping per minutes
MINUTES=0
LIMIT=${TIMEOUT}
while kill -0 ${cmd_pid} >/dev/null 2>&1;
do
    echo -n -e " \b"  # never leave evidence

    if [ ${MINUTES} == ${LIMIT} ]; then
        break;
    fi

    MINUTES=$((MINUTES + 1))

    sleep 60
done
# return exit code of background process
wait ${cmd_pid}
