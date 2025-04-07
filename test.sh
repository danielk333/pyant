#!/bin/ash

for ver in "3.7" "3.8" "3.9" "3.10" "3.11" "3.12" "3.13"
do
    uv venv -p "$ver" --seed "venv$ver"
    source "venv$ver/bin/activate"
    pip install .
    pip install pytest
    pytest
done
