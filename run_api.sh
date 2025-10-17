#!/bin/bash
cd /home/runner/workspace
export PYTHONPATH=/home/runner/workspace/src:$PYTHONPATH
exec uvicorn neurosymbolic_codegen.api.simple_main:app --host 0.0.0.0 --port 5000 --reload
