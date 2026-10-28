# Lichess bot image: rustfish behind the official lichess-bot bridge.
# Built by Dokku from the repo root (see deploy/lichess-bot/README.md).

FROM rust:1-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Dokku builds on the host that runs the bot, so native codegen is safe.
RUN RUSTFLAGS="-C target-cpu=native" cargo build --release

FROM python:3.12-slim
# Pinned lichess-bot master commit (2026-10-27); bump deliberately.
ARG LICHESS_BOT_SHA=afa430da25cb6d506c878d3ca2fa155be8da625d
WORKDIR /app
RUN python -c "import io, tarfile, urllib.request; \
url = 'https://github.com/lichess-bot-devs/lichess-bot/archive/${LICHESS_BOT_SHA}.tar.gz'; \
tar = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())); \
root = tar.getmembers()[0].name.split('/')[0]; \
[setattr(m, 'name', m.name.split('/', 1)[1]) for m in tar.getmembers() if '/' in m.name]; \
tar.extractall('/app', members=[m for m in tar.getmembers() if m.name != root], filter='data')" \
 && pip install --no-cache-dir -r requirements.txt
COPY --from=build /src/target/release/rustfish /app/engines/rustfish
COPY deploy/lichess-bot/config.yml /app/config.yml
# Same relative path as in the repo, so Dokku's procfile-path resolves whether
# it reads the Procfile from the repo or from the image WORKDIR.
COPY deploy/lichess-bot/Procfile /app/deploy/lichess-bot/Procfile
ENV LICHESS_BOT_DOCKER=1 PYTHONUNBUFFERED=1
CMD ["python", "lichess-bot.py", "--config", "config.yml"]
