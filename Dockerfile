FROM python:3.9

WORKDIR /segmentation

COPY . /segmentation

RUN apt-get update && apt-get install -y docker.io
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get --yes install libgl1

WORKDIR /segmentation
# Точка входа для контейнера
ENTRYPOINT ["python", "final_entrypoint.py"]
