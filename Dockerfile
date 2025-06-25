FROM python:3.9-slim
RUN apt-get update && apt-get install -y procps
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY ./api /app/api
COPY ./model /app/model
COPY ./data /app/data
CMD ["gunicorn", "api.app:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "4", "--bind", "0.0.0.0:8000", "--graceful-timeout", "30"]