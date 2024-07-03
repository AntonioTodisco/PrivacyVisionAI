#!/bin/bash

# Get setup script abspath.
script_path=$(readlink -f "$0")

# Get dirname from script abspath.
script_dir=$(dirname "$script_path")

# Set virtual environment name.
env_name="env"

# Get env name.
env_path="${script_dir}/${env_name}"

echo "[START] Start 'PrivacyVisionAI'..."
${env_path}/bin/python3 $script_dir/GUI.py
