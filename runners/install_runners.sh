#!/usr/bin/env bash
set -eo pipefail
set -x
# Run this script to set up symlinks that can be double clicked in the desktop

PARENT=/home/dieter/Desktop
RUNNER_SYMLINK_FOLDER=${PARENT}
SOURCE_SCRIPT_FOLDER="$(dirname $(realpath $0))"

if [ ! -d ${PARENT} ]; then
    echo "Parent folder ${PARENT} does not exist - did you specify the right path?"
    exit 1
fi

mkdir -p ${RUNNER_SYMLINK_FOLDER}
for fpath in $(find ${SOURCE_SCRIPT_FOLDER} -name 'Run*.sh'); do
    link_path="${RUNNER_SYMLINK_FOLDER}/$(basename $fpath)"
    if [ ! -e $link_path ]; then
        ln -s "$fpath" "$link_path"
        echo "Linked $fpath => $link_path"
        chmod +x "$link_path"
    fi
done