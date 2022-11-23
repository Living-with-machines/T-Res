FROM python:3.9

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY app/deezy_mostpopular_app.py /app/
CMD ["poetry", "run", "uvicorn", "deezy_mostpopular_app:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]
