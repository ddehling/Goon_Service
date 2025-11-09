#!/bin/bash
# Wrapper for running python scripts so we can run validation and error checking in one place
# in other files, then just can call ${PARENT_DIR}/run_python_script.sh GS_full.py ... etc
set -eo pipefail
set -x # REMOVE ME AFTER TESTING DONE
PARENT_DIR="$(dirname $(realpath $0)))"
SELECTED_FILE="$1"

function checkArguments {
    if [ "$#" -ne 1 ]; then
        echo "Must specify one and only one argument (the file to run)"
        exit 1
    fi
}

function runGoon() {
    echo "Gooooonnnnnnnnnn starting"
    checkArguments
    cd ${PARENT_DIR}
    python ${PARENT_DIR}/${SELECTED_FILE}

}
function reportFailure() {
    echo "gooning failed"
    exit 1
}
runGoon || reportFailure