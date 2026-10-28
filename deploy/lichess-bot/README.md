# rustfish Lichess bot (Dokku)

rustfish runs behind the official [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)
bridge as a `worker` process on the Dokku host `dev.duk.im` (app `rustfish-bot`).
It accepts bot and human challenges and, when idle, challenges other bots
to rated 3+2 / 5+3 games.

The Lichess bot rating lives in Lichess's bot pool. It does not convert to CCRL;
use the `CCRL ladder` workflow for the real strength number.

## One-time setup
```bash
ssh dokku@dev.duk.im apps:create rustfish-bot
ssh dokku@dev.duk.im builder-dockerfile:set rustfish-bot dockerfile-path deploy/lichess-bot/Dockerfile
ssh dokku@dev.duk.im ps:set rustfish-bot procfile-path deploy/lichess-bot/Procfile
ssh dokku@dev.duk.im resource:limit rustfish-bot --cpu 1 --memory 768m
ssh dokku@dev.duk.im checks:disable rustfish-bot
ssh dokku@dev.duk.im proxy:disable rustfish-bot
ssh dokku@dev.duk.im config:set --no-restart rustfish-bot LICHESS_BOT_TOKEN=<bot:play token>
git remote add dokku dokku@dev.duk.im:rustfish-bot
git push dokku main
ssh dokku@dev.duk.im ps:scale rustfish-bot web=0 worker=1
```

The Lichess account must be brand new (zero games played). Upgrade it to a bot
account once, after the token is set:
```bash
ssh dokku@dev.duk.im run rustfish-bot python lichess-bot.py --config config.yml -u
```

## Day to day
- Deploy a new engine version: `git push dokku main` (rebuilds rustfish on the server).
- Logs: `ssh dokku@dev.duk.im logs rustfish-bot -t`
- Pause: `ssh dokku@dev.duk.im ps:scale rustfish-bot worker=0`
