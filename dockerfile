FROM python:3.9.6-slim
WORKDIR /usr/src/app
COPY . ./

RUN apt update
RUN apt install -y python3-pip

RUN pip3 install gdown
RUN pip3 install -r req_bot.txt

RUN gdown --id 1heNu80X8DcTKTx2Od0-EW-6JrkXxk5Ze -O data/model.pt
RUN gdown --id 1c4LakbKi7-gbKyAvcoGkJ8Yic16wvJx0 -O data/ru_bpe.yttm
RUN gdown --id 1I46t9Qgz0NbXjT-EPbogEUYpvGPTc408 -O data/en_bpe.yttm

CMD ["python", "./bot.py"]