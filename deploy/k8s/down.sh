#!/usr/bin/env bash
# Delete the local kind cluster created by up.sh (volumes included).
set -euo pipefail
kind delete cluster --name "${CLUSTER:-finrag}"
