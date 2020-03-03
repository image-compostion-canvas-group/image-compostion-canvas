FROM triebwork/openpose-cpu-python:latest

COPY . /app

RUN pip3 install -r /app/requirements.txt

ENV IN_DIR="/images/in"
ENV OUT_DIR="/images/out"

ENTRYPOINT ["python3","/app/detect_structures.py"]