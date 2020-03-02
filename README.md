Important: replace all pathes with your own path to your local instalation. Always use absolute pathes to avoid problems.

## Installation
* Tested with `Python 3.7.5`, `pip 19.3.1` on `macOS 10.15`.
* Install and build OpenPose with python bindings (including OpenCV), for details see [official instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#installation).
* make sure to append the openpose python bindings to your python path: `export PYTHONPATH=/path/to/openpose/build/python:$PYTHONPATH`
* set env var for openpose models `export OPENPOSE_MODELS="/path/to/openpose/models/"`
* install all python dependencies: 'pip install -r requirements.txt'

## Usage
* make sure you have set all env vars from installation step
* prepare a folder with images as input folder
* change the flags in detect_structures.py to specify your output format
* start the main script, specify `IN_DIR` (folder with input images) and `OUT_DIR` for the results: `IN_DIR="/path/to/folder/with/input/images" OUT_DIR="/output/folder/with/results" python detect_structures.py`
