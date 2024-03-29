# TorchBot: A premitive toolkit for playing with pytorch models

## Features
- Supports single gpu, DataParallel and Distributed DataParallel.
- Supports logging to mlflow for tracking experiments.
- Model loading/saving.
- A bunch of utils for pytorch and python debugging.

## Requirements
See `requirements.txt` for a full list.

## Installation
 `python setup.py develop`


## Running

1. Start mlflow:
`sh examples/start_mlflow.sh`

2. Run your python file:
`python examples/example.py`


3. Check mlflow logs:
Open browser with `localhost:5000`


## Core Files
`torchbot/pipeline/engine.py`