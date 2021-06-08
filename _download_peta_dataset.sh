#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd $DIR
# Download Amazon Books Dataset
mkdir -p lepar/datasets/
pushd lepar/datasets
gdown --id 1E00TwWAW5KZihJGtWTKZK26CkWpQwrry
unzip peta-release.zip
mv peta-release raw
rm -f peta-release.zip
popd
popd