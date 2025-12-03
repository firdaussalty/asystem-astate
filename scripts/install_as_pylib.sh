#!/bin/bash
set -e

mkdir -p /tmp/astate_egg /tmp/astate_build

cd "$(dirname "$0")/../python"
python -m pip install .