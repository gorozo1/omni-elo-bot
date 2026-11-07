# OMNI TTS ELO Evaluation Bot

Telegram bot for pairwise Vietnamese TTS evaluations. At task start, choose single-speaker comparison or two-speaker dialogue comparison. Both modes collect a reference voice (optional), a text or image prompt/script, and two recordings, then use OpenAI GPT Audio through OpenRouter.

## Requirements

- Python 3.15.2 or newer for the tested local build
- A Telegram bot token from BotFather
- An OpenRouter API key with access to `openai/gpt-audio`

## Run locally

```sh
python3.15 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `TELEGRAM_BOT_TOKEN` and `OPENROUTER_API_KEY` in `.env`, then load those variables into your shell and start the worker:

```sh
set -a && source .env && set +a
python bot.py
```

After `/start`, `/newtask`, or `new task`, choose an evaluation type. Both types then follow the same five-step sequence: reference voice or `none`, prompt text or screenshot, first recording, second recording, and evaluation. For dialogue mode, the prompt should identify Speaker 1 and Speaker 2 turns. Media is limited to 20 MB by default; adjust `MAX_MEDIA_BYTES` if needed for your Telegram/OpenRouter setup.

## Telegram commands

- `/start` or `/newtask` starts or resets your session and asks you to choose single-speaker or two-speaker dialogue evaluation.
- `/status` shows which input the bot is waiting for.
- `/cancel` discards your active session.
- `/help` lists available commands.
- `/myid` shows your Telegram user ID for configuring admin access.
- `/logs` or `/logs 100` shows recent logs to configured admins in a private chat.

These commands operate the bot only while its polling process is running. `/start` starts an evaluation session; it cannot start a stopped local or hosted bot process.

## Activity logs

The bot writes timestamped activity and exception logs to `LOG_FILE` (default `data/omni-elo-bot.log`) and rotates the file at 2 MB, keeping three backups by default. It records workflow events and evaluation outcomes but does not log prompt text or audio data.

In Telegram, send `/myid` to get your numeric Telegram user ID. Add that ID to `ADMIN_USER_IDS` in `.env` (multiple IDs may be comma-separated), then restart the bot. Configured admins can privately request `/logs` or `/logs 100` for up to 200 recent lines. Log retrieval is denied in group chats.

## Evaluation and Elo

Both evaluation modes send the prompt/script and base64-encoded MP3 audio to OpenRouter's `openai/gpt-audio`; the selected mode controls the evaluator instructions, JSON schema, labels, and Telegram report. Single-speaker mode evaluates recording quality, persona likeness, pronunciation/text faithfulness, naturalness, and preference. Two-speaker mode evaluates conversational responsiveness, conversational naturalness, whole-conversation preference, and independently classifies each clip as human or AI-generated. Audio is converted to MP3 before sending. Screenshots are transcribed by `OPENROUTER_VISION_MODEL` before either mode evaluates the recordings. The model is prompted for JSON; the response parser also handles fenced JSON because this model does not accept a `response_format` parameter.

For single-speaker tasks, a supplied reference recording is sent as the first audio part, explicitly labeled as mandatory persona evidence. The response schema allows `[N/A]` for Persona Likeness only when no reference was supplied. Evaluation-start logs record whether a reference is present and its byte count, never its contents.

Each pair is rated as an isolated duel, with both responses starting at 1000 and a configurable `ELO_K_FACTOR` (default 32). The bot does not assume that Response A or B identifies a persistent voice across tasks. The Elo helper can be adapted to persistent competitor IDs if the workflow later supplies those identities.

## Deployment

`Procfile` supports worker-based hosts. `render.yaml` defines a Render background worker and persistent disk for SQLite session data. On Railway, deploy as a worker using `python bot.py` and mount a persistent volume at the directory used by `DATABASE_PATH`; do not store credentials in source control.

The database stores in-progress session media so polling restarts do not discard a task. Restrict access to the host volume and configure a suitable retention policy for your data requirements.

## Tests

```sh
python -m unittest discover -s tests
```