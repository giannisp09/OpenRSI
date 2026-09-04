FROM ubuntu:24.04

RUN apt-get update && apt-get install -y python3 python3-pip netcat-openbsd curl && rm -rf /var/lib/apt/lists/*

COPY server.py /app/server.py
WORKDIR /app

EXPOSE 8080

CMD ["python3", "server.py"]
