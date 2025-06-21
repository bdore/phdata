FROM python:3.9-slim
WORKDIR /app
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./api /app/api
COPY ./model /app/model
COPY ./data /app/data
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]