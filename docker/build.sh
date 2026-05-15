#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DOCKER_BUILDKIT=0 docker build -f docker/Dockerfile -t "${IMAGE_TAG}" . || exit $?

#    --ssh default \

# After bind-mounting the repo in run.sh, refresh the editable install if needed:
#   ./docker/run.sh pip install -e . --no-deps
