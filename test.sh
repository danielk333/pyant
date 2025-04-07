#!/bin/ash

for ver in "3.7.9" "3.8.20" "3.9.21" "3.10.16" "3.11.11" "3.12.9" "3.13.2"
do
    uv venv -p "$ver" --seed "venv$ver"
    source "venv$ver/bin/activate"
    pip install .
    pip install pytest
    pytest
done
