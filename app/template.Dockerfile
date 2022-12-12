FROM python:3.9
ARG APP_NAME

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENV APP_CONFIG_NAME=${APP_NAME}
COPY app/app_template.py /app/app.py
COPY app/configs/${APP_NAME}.py /app/config.py
CMD ["poetry", "run", "uvicorn", "app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]

#TODO: Use variable in Dockerfile