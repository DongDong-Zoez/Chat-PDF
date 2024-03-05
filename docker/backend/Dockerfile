FROM ubuntu

RUN apt-get update && \
    apt-get install -y apt-transport-https software-properties-common && \
    #  python3.9 python3-pip
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip

WORKDIR /app
COPY requirements.txt .
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.9 /usr/bin/python
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3

RUN pip3 install -r requirements.txt
RUN pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install gunicorn==19.9.0

ADD ./backend /app/backend
ADD ./docker /app/docker
ADD ./frontend /app/frontend

RUN mkdir -p /app/backend/server/static
RUN chmod +x /app/docker/backend/wsgi-entrypoint.sh