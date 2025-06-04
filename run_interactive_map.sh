#!/bin/bash
# This script starts the Flask server for the interactive station map

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install required packages if not already installed
pip install flask

# Start the Flask server
echo "Starting Flask server..."
echo "The interactive map will be available at http://localhost:8082"
python src/web/app.py

# Deactivate virtual environment when done
if [ -d "venv" ]; then
    deactivate
fi
