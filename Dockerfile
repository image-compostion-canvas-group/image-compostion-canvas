FROM openpose-cpu-python:latest

COPY . /app

RUN pip3 install -r /app/requirements.txt

ENV IN_DIR="/images/in"
ENV OUT_DIR="/images/out"

ENTRYPOINT ["python3","/app/detect_structures.py"]
# docker run --rm -it -v "/Users/Tilman/Documents/Programme/Python/forschungspraktikum/art-structures-env/src/images/images_imdahl/:/images/in" -v "/Users/Tilman/Documents/Programme/Python/forschungspraktikum/art-structures-env/src/images/out/docker_out/:/images/out" image-composition-canvas