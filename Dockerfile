# Dependencies stage (cached until requirements.txt changes)
FROM python:3.10 as dependencies

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Application stage (rebuilds on code changes)
FROM python:3.10

WORKDIR /code

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY ./src /code/src
COPY ./main.py /code/

EXPOSE 8002

CMD ["python", "-m", "main", "--port", "8002"]
