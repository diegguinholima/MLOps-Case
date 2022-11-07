FROM python:3.9

ADD . /app

WORKDIR /app

COPY . /app 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD [ "python", "./train.py"]