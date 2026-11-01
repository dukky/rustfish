# rustfish progress dashboard: stdlib-only Python, no pip installs.
# Built by Dokku from the repo root (see deploy/dashboard/README.md).
FROM python:3.12-slim
WORKDIR /app
COPY deploy/dashboard/server.py deploy/dashboard/index.html ./
# Same relative path as in the repo, so Dokku's procfile-path resolves.
COPY deploy/dashboard/Procfile /app/deploy/dashboard/Procfile
ENV PYTHONUNBUFFERED=1 PORT=5000
EXPOSE 5000
USER nobody
CMD ["python", "server.py"]
