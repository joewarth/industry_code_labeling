FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api /app/api
COPY training/artifacts /app/training/artifacts

EXPOSE 7860

CMD ["python", "-u", "api/app.py"]