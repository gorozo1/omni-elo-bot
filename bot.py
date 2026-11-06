from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

import telebot
import imageio_ffmpeg
from openai import OpenAI

LOG_FILE = os.getenv("LOG_FILE", "data/omni-elo-bot.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(2 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "3"))
MAX_LOG_LINES = 200
ADMIN_USER_IDS = {
    int(user_id.strip())
    for user_id in os.getenv("ADMIN_USER_IDS", "").split(",")
    if user_id.strip().isdigit()
}


def configure_logging() -> None:
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not any(isinstance(handler, RotatingFileHandler) for handler in root_logger.handlers):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in root_logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


def read_log_tail(path: str | Path, line_count: int) -> list[str]:
    if line_count <= 0:
        return []
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as log_file:
            return list(deque(log_file, maxlen=line_count))
    except FileNotFoundError:
        return []


configure_logging()
logger = logging.getLogger("omni_elo_bot")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-audio")
VISION_MODEL_NAME = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
MAX_MEDIA_BYTES = int(os.getenv("MAX_MEDIA_BYTES", str(20 * 1024 * 1024)))
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/sessions.sqlite3")
WAIT_REFERENCE = "wait_reference"
WAIT_PROMPT = "wait_prompt"
WAIT_RESPONSE_A = "wait_response_a"
WAIT_RESPONSE_B = "wait_response_b"

BOT_COMMANDS = (
    ("start", "Start an evaluation"),
    ("newtask", "Reset and start a new evaluation"),
    ("status", "Show the current evaluation step"),
    ("cancel", "Cancel the current evaluation"),
    ("myid", "Show your Telegram user ID"),
    ("logs", "Show recent logs (admins only)"),
    ("help", "List bot commands"),
)

PAIRWISE_VERDICTS = ["[Response A]", "[Response B]", "[Both Good]", "[Both Bad]"]

SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "audio_recording_quality_verdict": {"type": "string", "enum": PAIRWISE_VERDICTS},
        "audio_recording_quality_breakdown": {"type": "string"},
        "persona_likeness_verdict": {"type": "string", "enum": ["[N/A]", *PAIRWISE_VERDICTS]},
        "persona_likeness_breakdown": {"type": "string"},
        "pronunciation_faithfulness_verdict": {"type": "string", "enum": PAIRWISE_VERDICTS},
        "pronunciation_breakdown": {"type": "string"},
        "technical_string_verdict": {
            "type": "string",
            "enum": ["[N/A]", *PAIRWISE_VERDICTS],
        },
        "technical_string_details": {"type": "string"},
        "naturalness_verdict": {"type": "string", "enum": ["[Response A]", "[Response B]"]},
        "naturalness_breakdown": {"type": "string"},
        "summary": {"type": "string"},
        "winner": {"type": "string", "enum": ["A", "B"]},
        "dominant_factor": {
            "type": "string",
            "enum": [
                "Audio and Recording Quality",
                "Persona Likeness",
                "Pronunciation Faithfulness",
                "Naturalness",
            ],
        },
    },
    "required": [
        "audio_recording_quality_verdict",
        "audio_recording_quality_breakdown",
        "persona_likeness_verdict",
        "persona_likeness_breakdown",
        "pronunciation_faithfulness_verdict",
        "pronunciation_breakdown",
        "technical_string_verdict",
        "technical_string_details",
        "naturalness_verdict",
        "naturalness_breakdown",
        "summary",
        "winner",
        "dominant_factor",
    ],
}


@dataclass
class Session:
    state: str
    reference_audio: bytes | None = None
    reference_mime: str | None = None
    prompt_text: str | None = None
    prompt_image: bytes | None = None
    prompt_image_mime: str | None = None
    response_a: bytes | None = None
    response_a_mime: str | None = None
    response_b: bytes | None = None
    response_b_mime: str | None = None


class SessionStore:
    """SQLite-backed sessions survive bot restarts and concurrent updates."""

    def __init__(self, path: str):
        self.path = path
        self.lock = threading.RLock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    user_id INTEGER PRIMARY KEY,
                        payload TEXT NOT NULL,
                        reference_audio BLOB,
                        prompt_image BLOB,
                        response_a BLOB,
                        response_b BLOB
                )"""
            )

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def get(self, user_id: int) -> Session | None:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload, reference_audio, prompt_image, response_a, response_b "
                "FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload"])
        payload.update(
            reference_audio=row["reference_audio"],
            prompt_image=row["prompt_image"],
            response_a=row["response_a"],
            response_b=row["response_b"],
        )
        return Session(**payload)

    def put(self, user_id: int, session: Session) -> None:
        payload = json.dumps(
            {
                "state": session.state,
                "reference_mime": session.reference_mime,
                "prompt_text": session.prompt_text,
                "prompt_image_mime": session.prompt_image_mime,
                "response_a_mime": session.response_a_mime,
                "response_b_mime": session.response_b_mime,
            }
        )
        with self.lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (user_id, payload, reference_audio, prompt_image, response_a, response_b) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload, "
                "reference_audio = excluded.reference_audio, prompt_image = excluded.prompt_image, "
                "response_a = excluded.response_a, response_b = excluded.response_b",
                (
                    user_id,
                    payload,
                    session.reference_audio,
                    session.prompt_image,
                    session.response_a,
                    session.response_b,
                ),
            )

    def delete(self, user_id: int) -> None:
        with self.lock, self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def escape_html(text: str) -> str:
    return html.escape(str(text), quote=False)


def is_greeting(text: str) -> bool:
    normalized = re.sub(r"[^\w]+", " ", text.casefold()).strip()
    greetings = (
        "hi",
        "hello",
        "hey",
        "hiya",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "xin chào",
        "chào",
    )
    if normalized in greetings or normalized == "chào bạn":
        return True
    return normalized.startswith(tuple(f"{greeting} " for greeting in greetings))


def score_prompt() -> str:
    return (
        "You are the OMNI TTS ELO evaluator. Strictly compare two Vietnamese TTS recordings against the supplied "
        "prompt text or screenshot and optional reference voice. Listen for every noticeable error in both clips; "
        "do not guess beyond audible evidence. Return all verdicts and concise, concrete breakdowns in English.\n\n"
        "VERDICT RULES\n"
        "1. Audio and Recording Quality: choose exactly [Response A], [Response B], [Both Good], or [Both Bad]. "
        "Penalize background noise, pops, static, fuzz, clipping, harsh compression, robotic echo, room reverb, "
        "and unusually long ending silence (over 2 seconds). Slurring or distortion lasting 3-5 seconds or "
        "10-25% of a clip is a serious failure.\n"
        "2. Persona Likeness: choose [N/A] when the reference is none; otherwise choose one pairwise verdict. "
        "Compare voice identity, perceived age and gender, accent, pitch, and timbre to the reference. Explicit "
        "gender or strong regional accent mismatch is a major failure. Do not confuse emotional variation with "
        "a different voice identity. Strong persona adherence beats merely higher naturalness.\n"
        "3. Pronunciation Faithfulness: compare spoken content to the prompt, including Vietnamese words, tone "
        "marks, numbers, times, units, punctuation meaning, omissions, additions, and hallucinated fillers. Judge "
        "speech clarity separately; clarity never excuses text errors. Expand numbers, times, and units naturally "
        "for Vietnamese rather than reading abbreviations character by character. Written lexical fillers must be "
        "spoken; added lexical fillers are alignment errors. Overused non-lexical fillers reduce naturalness.\n"
        "TECHNICAL STRING RULE: Treat IPv4/IPv6 addresses, URLs, file paths, codes, hashes, and similar strings "
        "as strict ordered tokens. Check every letter, digit, separator, and block. Missing, added, substituted, "
        "or garbled characters are severe alignment failures; explicitly spoken underscores and slashes must not "
        "be omitted when present in the prompt. Never call an altered token a pass. If both responses alter a "
        "technical string, set pronunciation_faithfulness_verdict and technical_string_verdict to [Both Bad]. "
        "If no technical string is present, use [N/A] for technical_string_verdict.\n"
        "4. Naturalness: assess human rhythm, pitch variation, expressive tone, breathing pauses, sentence "
        "spacing, and robotic or repetitive cadence. A natural-sounding clip must not win over a more faithful "
        "clip when text accuracy is the decisive difference. Naturalness verdict must be Response A or B; never tie.\n"
        "5. Overall preference must be exactly A or B; ties are forbidden. Select the single dominant factor from "
        "the four listed factors and make it the most important reason for the winner. Explain any trade-off.\n\n"
        "Each breakdown must be 1-2 concise sentences and mention relevant evidence for both responses. Do not "
        "use vague phrases such as 'B is better', 'sounds good', or 'A is bad'. The final summary is the copy-ready "
        "justification: 15-25 words and at least 100 characters, naming the winner, dominant factor, concrete "
        "flaws in both clips, and any trade-off. Use plain language.\n\n"
        "VERIFIED COMPARISON EXAMPLES\n"
        "- A has stiff repetitive cadence and poor sentence pauses while B is more natural: winner B, dominant "
        "factor Naturalness.\n"
        "- A strongly expresses the reference persona while B is flatter despite sounding conventionally natural: "
        "winner A, dominant factor Persona Likeness.\n"
        "- Reward a distinctive persona even when its delivery is emotionally restrained, if the voice still matches "
        "the supplied persona/reference and is not stiff.\n"
        "Return only JSON matching the requested schema."
    )


def format_result(result: dict) -> str:
    def verdict(key: str, fallback: str = "[Both Good]") -> str:
        return str(result.get(key, fallback))

    def breakdown(key: str, metric: str) -> str:
        value = result.get(key)
        if value:
            return str(value)
        response_a = result.get("response_a", {}).get(metric)
        response_b = result.get("response_b", {}).get(metric)
        if response_a or response_b:
            return f"Response A: {response_a or 'no specific issue noted'}. Response B: {response_b or 'no specific issue noted'}."
        return str(result.get("summary", "No additional comparison details were provided."))

    winner = str(result["winner"])
    if winner not in {"A", "B"}:
        raise ValueError("winner must be A or B")
    pronunciation = breakdown("pronunciation_breakdown", "pronunciation")
    technical_verdict = verdict("technical_string_verdict", "[N/A]")
    technical_details = result.get("technical_string_details")
    if technical_details:
        pronunciation = f"{pronunciation} Technical string verdict: {technical_verdict}. {technical_details}"
    else:
        pronunciation = f"Technical string verdict: {technical_verdict}. {pronunciation}"

    lines = [
        "<b>🔍 Evaluation Breakdown</b>",
        "",
        f"<b>Audio and Recording Quality:</b> {escape_html(verdict('audio_recording_quality_verdict'))} - {escape_html(breakdown('audio_recording_quality_breakdown', 'audio_quality'))}",
        f"<b>Persona likeness (Speaker Similarity):</b> {escape_html(verdict('persona_likeness_verdict', '[N/A]'))} - {escape_html(breakdown('persona_likeness_breakdown', 'persona_likeness'))}",
        f"<b>Pronunciation Faithfulness:</b> {escape_html(verdict('pronunciation_faithfulness_verdict'))} - {escape_html(pronunciation)}",
        f"<b>Naturalness:</b> {escape_html(verdict('naturalness_verdict', '[Response A]'))} - {escape_html(breakdown('naturalness_breakdown', 'naturalness'))}",
        "",
        "<b>🏆 Final Verdict</b>",
        f"<b>Overall Preference:</b> Response {winner}",
        f"<b>Dominant Factor:</b> {escape_html(result['dominant_factor'])}",
        "",
        "<b>✍️ Justification (Ready to Copy-Paste)</b>",
        f'"{escape_html(result["summary"])}"',
    ]
    return "\n".join(lines)


def audio_input_part(audio_data: bytes) -> dict:
    try:
        conversion = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "96k",
                "-f",
                "mp3",
                "pipe:1",
            ],
            input=audio_data,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Could not convert this audio to the MP3 format required for evaluation.") from error

    encoded_audio = base64.b64encode(conversion.stdout).decode("ascii")
    return {
        "type": "input_audio",
        "input_audio": {"data": encoded_audio, "format": "mp3"},
    }


def build_messages(session: Session) -> list[dict]:
    user_content = [{"type": "text", "text": f"Prompt text:\n{session.prompt_text or '(no text prompt provided)'}"}]
    if session.reference_audio is None:
        user_content.append({"type": "text", "text": "Reference voice: none was provided."})
    else:
        user_content.extend(
            [
                {"type": "text", "text": "Reference voice audio:"},
                audio_input_part(session.reference_audio),
            ]
        )
    user_content.extend(
        [
            {"type": "text", "text": "Response A audio:"},
            audio_input_part(session.response_a),
            {"type": "text", "text": "Response B audio:"},
            audio_input_part(session.response_b),
        ]
    )
    system_prompt = (
        score_prompt()
        + "\n\nReturn a JSON object following this schema:\n"
        + json.dumps(SCORING_SCHEMA, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def extract_prompt_from_image(client: OpenAI, session: Session) -> str:
    if session.prompt_image is None:
        return session.prompt_text or ""
    image_data = base64.b64encode(session.prompt_image).decode("ascii")
    image_url = f"data:{session.prompt_image_mime or 'image/jpeg'};base64,{image_data}"
    response = client.chat.completions.create(
        model=VISION_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Transcribe the prompt text visible in this screenshot exactly. Preserve Vietnamese diacritics, "
                    "punctuation, numbers, and technical strings. Do not summarize or add commentary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the complete text prompt from this screenshot."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0,
        max_tokens=1500,
    )
    extracted_text = response.choices[0].message.content
    if not extracted_text:
        raise RuntimeError("OpenRouter returned no text for the prompt screenshot.")
    return "\n".join(part.text for part in extracted_text if getattr(part, "text", None)) if isinstance(extracted_text, list) else extracted_text.strip()


def get_audio(bot, message) -> tuple[bytes, str]:
    media = message.voice or message.audio
    if not media:
        raise ValueError("Please send an audio or voice message.")
    if media.file_size and media.file_size > MAX_MEDIA_BYTES:
        raise ValueError(f"Audio exceeds the {MAX_MEDIA_BYTES // (1024 * 1024)} MB limit.")
    file_info = bot.get_file(media.file_id)
    data = bot.download_file(file_info.file_path)
    if len(data) > MAX_MEDIA_BYTES:
        raise ValueError(f"Audio exceeds the {MAX_MEDIA_BYTES // (1024 * 1024)} MB limit.")
    return data, (media.mime_type or ("audio/ogg" if message.voice else "audio/mpeg"))


def get_photo(bot, message) -> tuple[bytes, str]:
    photo = message.photo[-1]
    if photo.file_size and photo.file_size > MAX_MEDIA_BYTES:
        raise ValueError(f"Image exceeds the {MAX_MEDIA_BYTES // (1024 * 1024)} MB limit.")
    file_info = bot.get_file(photo.file_id)
    data = bot.download_file(file_info.file_path)
    if len(data) > MAX_MEDIA_BYTES:
        raise ValueError(f"Image exceeds the {MAX_MEDIA_BYTES // (1024 * 1024)} MB limit.")
    return data, "image/jpeg"


def create_bot():
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not telegram_token or not openrouter_key:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and OPENROUTER_API_KEY before starting the bot.")

    telegram = telebot.TeleBot(telegram_token, threaded=True)
    client = OpenAI(
        api_key=openrouter_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={"X-Title": "OMNI TTS ELO"},
    )
    store = SessionStore(DATABASE_PATH)
    user_locks: dict[int, threading.Lock] = {}
    user_locks_guard = threading.Lock()

    def lock_for(user_id: int):
        with user_locks_guard:
            return user_locks.setdefault(user_id, threading.Lock())

    def begin_task(message, source: str):
        store.put(message.from_user.id, Session(state=WAIT_REFERENCE))
        logger.info("Task started user_id=%s source=%s", message.from_user.id, source)
        telegram.reply_to(
            message,
            "Ready for the next evaluation. Please send or record the *Reference Voice* (or type 'none' if there is no reference).",
            parse_mode="Markdown",
        )

    @telegram.message_handler(commands=["start", "newtask"])
    def start_command(message):
        begin_task(message, "command")

    @telegram.message_handler(commands=["help"])
    def help_command(message):
        help_text = (
            "Bot commands:\n"
            "/start - Start an evaluation\n"
            "/newtask - Reset and start a new evaluation\n"
            "/status - Show the current evaluation step\n"
            "/cancel - Cancel the current evaluation\n"
            "/myid - Show your Telegram user ID\n"
            "/logs [1-200] - Show recent logs (configured admins, private chat only)\n"
            "/help - Show this command list\n\n"
            "The bot process itself must be running in its terminal or hosting service."
        )
        telegram.reply_to(message, help_text)

    @telegram.message_handler(commands=["status"])
    def status_command(message):
        session = store.get(message.from_user.id)
        if session is None:
            telegram.reply_to(message, "No active evaluation. Send /newtask to begin.")
            return
        status_by_state = {
            WAIT_REFERENCE: "Waiting for the reference voice, or `none`.",
            WAIT_PROMPT: "Waiting for prompt text or a screenshot.",
            WAIT_RESPONSE_A: "Waiting for Response A audio.",
            WAIT_RESPONSE_B: "Waiting for Response B audio.",
        }
        telegram.reply_to(message, status_by_state.get(session.state, "Evaluation state is unknown."))

    @telegram.message_handler(commands=["cancel"])
    def cancel_command(message):
        had_active_task = store.get(message.from_user.id) is not None
        store.delete(message.from_user.id)
        logger.info("Task cancelled user_id=%s active=%s", message.from_user.id, had_active_task)
        telegram.reply_to(message, "Task cancelled. Send *new task* when you are ready.", parse_mode="Markdown")

    @telegram.message_handler(commands=["myid"])
    def my_id_command(message):
        telegram.reply_to(message, f"Your Telegram user ID is {message.from_user.id}.")

    @telegram.message_handler(commands=["logs"])
    def logs_command(message):
        user_id = message.from_user.id
        if not ADMIN_USER_IDS:
            telegram.reply_to(message, "Log access is not configured. Set ADMIN_USER_IDS in the bot environment.")
            return
        if message.chat.type != "private" or user_id not in ADMIN_USER_IDS:
            logger.warning("Denied log access user_id=%s chat_type=%s", user_id, message.chat.type)
            telegram.reply_to(message, "Log access is restricted to configured admins in a private chat.")
            return

        parts = (message.text or "").split(maxsplit=1)
        try:
            line_count = int(parts[1]) if len(parts) > 1 else 60
        except ValueError:
            telegram.reply_to(message, "Usage: /logs [1-200]")
            return
        if not 1 <= line_count <= MAX_LOG_LINES:
            telegram.reply_to(message, "Choose a line count between 1 and 200.")
            return

        lines = read_log_tail(LOG_FILE, line_count)
        if not lines:
            telegram.reply_to(message, "No log entries yet.")
            return

        messages = []
        current = f"Recent bot logs (up to {line_count} lines):\n"
        for line in lines:
            if len(current) + len(line) > 3500:
                messages.append(current)
                current = ""
            current += line[-3400:] if len(line) > 3400 else line
        if current:
            messages.append(current)
        logger.info("Logs requested user_id=%s lines=%s", user_id, len(lines))
        for log_message in messages:
            telegram.send_message(message.chat.id, log_message, disable_notification=True)

    @telegram.message_handler(content_types=["text", "audio", "voice", "photo"])
    def handle_message(message):
        user_id = message.from_user.id
        with lock_for(user_id):
            text = (message.text or "").strip()
            if text.casefold() == "new task" or is_greeting(text):
                begin_task(message, "text")
                return

            session = store.get(user_id)
            if session is None:
                telegram.reply_to(message, "Send *new task* to start an evaluation.", parse_mode="Markdown")
                return

            try:
                if session.state == WAIT_REFERENCE:
                    if text.casefold() == "none":
                        session.reference_audio = None
                        session.reference_mime = None
                        reference_description = "none"
                    else:
                        session.reference_audio, session.reference_mime = get_audio(telegram, message)
                        reference_description = f"audio bytes={len(session.reference_audio)}"
                    session.state = WAIT_PROMPT
                    store.put(user_id, session)
                    logger.info("Reference received user_id=%s kind=%s", user_id, reference_description)
                    telegram.reply_to(
                        message,
                        "Reference logged. Please type the *Prompt Text* or upload a *Screenshot* of the prompt page.",
                        parse_mode="Markdown",
                    )
                elif session.state == WAIT_PROMPT:
                    if message.photo:
                        session.prompt_image, session.prompt_image_mime = get_photo(telegram, message)
                        session.prompt_text = message.caption
                        prompt_description = f"image bytes={len(session.prompt_image)} caption_chars={len(message.caption or '')}"
                    elif text:
                        session.prompt_text = text
                        prompt_description = f"text chars={len(text)}"
                    else:
                        raise ValueError("Please send prompt text or a screenshot.")
                    session.state = WAIT_RESPONSE_A
                    store.put(user_id, session)
                    logger.info("Prompt received user_id=%s kind=%s", user_id, prompt_description)
                    telegram.reply_to(message, "Prompt logged. Please send or record *Response A*.", parse_mode="Markdown")
                elif session.state == WAIT_RESPONSE_A:
                    session.response_a, session.response_a_mime = get_audio(telegram, message)
                    session.state = WAIT_RESPONSE_B
                    store.put(user_id, session)
                    logger.info("Response A received user_id=%s bytes=%s mime=%s", user_id, len(session.response_a), session.response_a_mime)
                    telegram.reply_to(message, "Response A logged. Please send or record *Response B*.", parse_mode="Markdown")
                elif session.state == WAIT_RESPONSE_B:
                    session.response_b, session.response_b_mime = get_audio(telegram, message)
                    store.put(user_id, session)
                    logger.info("Response B received user_id=%s bytes=%s mime=%s", user_id, len(session.response_b), session.response_b_mime)
                    evaluation_started = time.monotonic()
                    logger.info("Evaluation started user_id=%s model=%s", user_id, MODEL_NAME)
                    if session.prompt_image is not None:
                        screenshot_text = extract_prompt_from_image(client, session)
                        if session.prompt_text:
                            session.prompt_text = f"{session.prompt_text}\n\nText recognized from screenshot:\n{screenshot_text}"
                        else:
                            session.prompt_text = screenshot_text
                        session.prompt_image = None
                        session.prompt_image_mime = None
                        store.put(user_id, session)
                        logger.info("Screenshot prompt extracted user_id=%s chars=%s", user_id, len(screenshot_text))
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=build_messages(session),
                        temperature=0.2,
                        max_tokens=2500,
                    )
                    response_text = response.choices[0].message.content
                    if not response_text:
                        raise RuntimeError("OpenRouter returned an empty evaluation.")
                    result = json.loads(response_text)
                    logger.info(
                        "Evaluation completed user_id=%s winner=%s dominant_factor=%s duration_seconds=%.2f",
                        user_id,
                        result.get("winner"),
                        result.get("dominant_factor"),
                        time.monotonic() - evaluation_started,
                    )
                    telegram.send_message(
                        message.chat.id,
                        format_result(result),
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
                    store.delete(user_id)
            except ValueError as error:
                telegram.reply_to(message, str(error))
            except Exception:
                logger.exception("Task failed for Telegram user %s", user_id)
                if session.state == WAIT_RESPONSE_B:
                    telegram.reply_to(message, "Evaluation failed. Send Response B again to retry, or use /cancel.")
                else:
                    telegram.reply_to(message, "I couldn't process that input. Please retry the current step.")

    try:
        telegram.set_my_commands(
            [telebot.types.BotCommand(command, description) for command, description in BOT_COMMANDS]
        )
    except Exception:
        logger.exception("Could not register Telegram command menu")

    return telegram


if __name__ == "__main__":
    bot = create_bot()
    logger.info("Starting Telegram polling with model %s", MODEL_NAME)
    bot.infinity_polling(skip_pending=True, timeout=30, long_polling_timeout=30)