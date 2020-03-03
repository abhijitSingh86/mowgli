FROM python:3.7
ENV APP_DIR=/usr/src/app/
ADD Pipfile Pipfile.lock setup.py README.md $APP_DIR
ADD mowgli $APP_DIR/mowgli/
RUN pip install pipenv
WORKDIR $APP_DIR
RUN pipenv install --system --deploy --ignore-pipfile
CMD gunicorn -w 4 -b 0.0.0.0:$PORT mowgli.infrastructure.endpoints:app
