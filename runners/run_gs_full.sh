#!/bin/bash

# just change the file here
set -eo pipefail
SCRIPT_DIR="$(dirname $(realpath $0))"
FILE_TO_RUN="GS_full.py

${PARENT_DIR}/run_python_script.sh $FILE_TO_RUN