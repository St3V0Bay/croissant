#!/bin/bash

sudo apt-get update
sudo apt-get install -y libgraphviz-dev

# Install dev packages.
pip3 install python/mlcroissant/.[dev]
