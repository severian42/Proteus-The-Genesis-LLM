#!/bin/bash
Xvfb :99 -screen 0 1024x768x24 &
sleep 1
python -m uvicorn llm_physics_runner:app --host 0.0.0.0 --port 4242 