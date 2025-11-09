#!/usr/bin/env bash

# just change the file here
set -eo pipefail
set -x
SCRIPT_DIR="$(dirname $(realpath $0))"
FILE_TO_RUN="align_tool.py"

# x-terminal-emulator causes this to pop up in new window so we can quit
x-terminal-emulator -e "${SCRIPT_DIR}/run_python_script.sh $FILE_TO_RUN"