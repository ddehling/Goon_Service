#!/usr/bin/env bash
# Wrapper for running python scripts so we can run validation and error checking in one place
# in other files, then just can call ${PARENT_DIR}/run_python_script.sh GS_full.py ... etc
# probably not necessary but I can imagine some future python environment set up that's nicer to centralize.
# Expectation is you'd symlink the other files to the desktop.
set -eo pipefail
set -x # REMOVE ME AFTER TESTING DONE
PARENT_DIR="$(dirname $(dirname $(realpath $0)))"
PYTHON="${PARENT_DIR}/env/bin/python"
SELECTED_FILE="$1"

# check arguments
if [ "$#" -ne 1 ]; then
    echo "$0: Must specify one and only one argument (the file to run)"
    exit 1
fi


function runGoon() {
    echo "Gooooonnnnnnnnnn starting"
    cd ${PARENT_DIR}
    "${PYTHON}" "${PARENT_DIR}/${SELECTED_FILE}"

}
function reportFailure() {
    echo "gooning failed"
    exit 1
}
runGoon || reportFailure