FROM python:3.11.4-bookworm

ADD . root/app
WORKDIR root/app


RUN pip install -r requirements.txt
# CMD python app.py

COPY ./app /root/app/
CMD tail -f /dev/null
