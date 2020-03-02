Important: replace all pathes with your own path to your local instalation. Always use absolute pathes to avoid problems.

## Installation
Tested with `Python 3.7.5`, `pip 19.3.1` on `macOS 10.15`.

* Optional: Create a virtual environment, to avoid version conflicts with previously installed packages:
```bash
git clone https://github.com/image-compostion-canvas-group/image-compostion-canvas
cd image-compostion-canvas
virtualenv .venv
source .venv/bin/activate
``` 
* If you skip the virtual env. Clone the repo with `git clone https://github.com/image-compostion-canvas-group/image-compostion-canvas`
* Install and build OpenPose with python bindings (including OpenCV), for details see [official instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md#installation).
* make sure to append the openpose python bindings to your python path and to set env var for openpose models. Make sure to replace `/path/to/openpose` with the absolute path to your openpose installation from the step above. Keep `/build/python` and `/models/` in this path.
```bash
export PYTHONPATH="/path/to/openpose/build/python:$PYTHONPATH"
export OPENPOSE_MODELS="/path/to/openpose/models/"
```
* install all python dependencies:
```bash
pip install -r requirements.txt
```

## Usage
* make sure you have set all env vars from installation step
* prepare a folder with images as input folder
* change the flags in detect_structures.py to specify your output format
* start the main script, specify `IN_DIR` (folder with input images) and `OUT_DIR` for the results: `IN_DIR="/path/to/folder/with/input/images" OUT_DIR="/output/folder/with/results" python detect_structures.py`
