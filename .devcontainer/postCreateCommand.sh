#!/bin/bash

sudo apt-get update && sudo apt-get install -y graphviz libgraphviz-dev git-lfs

# Install dependencies except mlcroissant itself
# Use the workspace root to find the correct path
WORKSPACE_ROOT="${PWD}"
if [[ $PWD == *".devcontainer"* ]]; then
    WORKSPACE_ROOT="${PWD%/.devcontainer}"
fi

cd "${WORKSPACE_ROOT}/python/mlcroissant"
pip install -e .[dev,dicom]

# Install coverage separately since the feature might not work
pip install coverage
