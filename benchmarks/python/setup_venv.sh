#!/bin/bash
set -e

echo "🧹 Recreating clean virtual environment venv_clean..."
rm -rf venv_clean
python3 -m venv venv_clean

echo "🔄 Upgrading pip..."
venv_clean/bin/pip install --upgrade pip

echo "📦 Installing requirements..."
venv_clean/bin/pip install -r benchmarks/python/requirements.txt

echo "✅ Environment setup complete!"
