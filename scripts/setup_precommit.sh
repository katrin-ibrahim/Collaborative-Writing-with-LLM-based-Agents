#!/usr/bin/env bash

# Install or update pre-commit
pip install -U pre-commit

# Install the git hook scripts
pre-commit install

# Update to the latest versions
pre-commit autoupdate

echo "✅ Pre-commit hooks have been installed and updated successfully!"
echo "💡 These hooks will run automatically on git commit. To run them manually use: pre-commit run --all-files"
