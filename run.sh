#!/usr/bin/env bash
# Run the face tracker from anywhere. Usage: ./run.sh   or   bash run.sh

cd "$(dirname "$0")"
. .venv/bin/activate
exec python face_tracker.py "$@"
