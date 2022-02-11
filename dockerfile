FROM python:3.9.10-slim
WORKDIR /usr/src/app
COPY . ./
RUN ./dinstall.sh
CMD ["python", "./bot.py"]