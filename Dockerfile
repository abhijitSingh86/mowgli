FROM python:3.5
ENV APP_DIR=/usr/src/app/
WORKDIR $APP_DIR
ADD requirements.txt setup.py README.md $APP_DIR
ADD resources $APP_DIR/resources/
ADD mowgli $APP_DIR/mowgli/
ADD data $APP_DIR/data/
RUN pip install -r requirements.txt
RUN useradd -M -s /bin/sh mowgli && chown -R mowgli:mowgli $APP_DIR
USER mowgli
CMD gunicorn -w 4 -b 0.0.0.0:$PORT mowgli.infrastructure.endpoints:APP
