import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bot import (
    DIALOGUE_SCHEMA,
    EVALUATION_DIALOGUE,
    EVALUATION_SINGLE,
    PAIRWISE_VERDICTS,
    Session,
    SessionStore,
    audio_input_part,
    build_messages,
    extract_prompt_from_image,
    format_result,
    parse_model_json,
)


class OpenRouterPayloadTests(unittest.TestCase):
    def test_parser_accepts_json_fenced_in_markdown(self):
        self.assertEqual(parse_model_json('```json\n{"winner":"A"}\n```'), {"winner": "A"})

    def test_parser_accepts_json_surrounded_by_explanatory_text(self):
        self.assertEqual(parse_model_json('Result follows:\n{"winner":"B"}\nDone.'), {"winner": "B"})

    def test_existing_sessions_default_to_single_speaker(self):
        self.assertEqual(Session(state="wait_reference").evaluation_type, EVALUATION_SINGLE)

    def test_dialogue_type_persists_in_session_store(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as directory:
            store = SessionStore(str(Path(directory) / "sessions.sqlite3"))
            store.put(123, Session(state="wait_prompt", evaluation_type=EVALUATION_DIALOGUE))

            self.assertEqual(store.get(123).evaluation_type, EVALUATION_DIALOGUE)

    def test_dialogue_schema_has_all_requested_dimensions(self):
        self.assertEqual(
            DIALOGUE_SCHEMA["properties"]["overall_preference"]["enum"],
            ["[Conversation A]", "[Conversation B]", "[Both Good]", "[Both Bad]"],
        )
        self.assertEqual(
            DIALOGUE_SCHEMA["properties"]["conversation_a_classification"]["enum"],
            ["Human recording", "AI-generated"],
        )

    @patch("bot.subprocess.run")
    @patch("bot.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    def test_audio_is_converted_and_base64_encoded(self, get_ffmpeg, run):
        run.return_value = SimpleNamespace(stdout=b"mp3-audio")

        part = audio_input_part(b"telegram-ogg")

        self.assertEqual(part["type"], "input_audio")
        self.assertEqual(part["input_audio"]["format"], "mp3")
        self.assertEqual(base64.b64decode(part["input_audio"]["data"]), b"mp3-audio")
        self.assertEqual(run.call_args.kwargs["input"], b"telegram-ogg")

    @patch("bot.subprocess.run")
    @patch("bot.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    def test_message_builder_includes_all_audio_inputs(self, get_ffmpeg, run):
        run.return_value = SimpleNamespace(stdout=b"converted-mp3")
        session = Session(
            state="evaluate",
            prompt_text="Xin chao",
            reference_audio=b"reference",
            response_a=b"audio-a",
            response_b=b"audio-b",
        )

        messages = build_messages(session)
        audio_parts = [part for part in messages[1]["content"] if part["type"] == "input_audio"]

        self.assertEqual(len(audio_parts), 3)
        self.assertTrue(any("Xin chao" in part.get("text", "") for part in messages[1]["content"]))

    @patch("bot.subprocess.run")
    @patch("bot.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    def test_reference_is_first_audio_and_required_for_persona_comparison(self, get_ffmpeg, run):
        run.return_value = SimpleNamespace(stdout=b"converted-mp3")
        session = Session(
            state="evaluate",
            prompt_text="Xin chao",
            reference_audio=b"reference-voice",
            response_a=b"audio-a",
            response_b=b"audio-b",
        )

        messages = build_messages(session)
        content = messages[1]["content"]
        request_schema = json.loads(
            messages[0]["content"].rsplit("Return a JSON object following this schema:\n", 1)[1]
        )

        self.assertIn("MANDATORY REFERENCE VOICE", content[0]["text"])
        self.assertEqual(content[1]["type"], "input_audio")
        self.assertIn("Prompt text:", content[2]["text"])
        self.assertEqual(request_schema["properties"]["persona_likeness_verdict"]["enum"], PAIRWISE_VERDICTS)

    @patch("bot.subprocess.run")
    @patch("bot.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    def test_no_reference_requires_not_applicable_persona_verdict(self, get_ffmpeg, run):
        run.return_value = SimpleNamespace(stdout=b"converted-mp3")
        session = Session(state="evaluate", prompt_text="Xin chao", response_a=b"audio-a", response_b=b"audio-b")

        messages = build_messages(session)
        request_schema = json.loads(
            messages[0]["content"].rsplit("Return a JSON object following this schema:\n", 1)[1]
        )

        self.assertIn("None provided", messages[1]["content"][0]["text"])
        self.assertEqual(request_schema["properties"]["persona_likeness_verdict"]["enum"], ["[N/A]"])

    @patch("bot.subprocess.run")
    @patch("bot.imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    def test_dialogue_mode_builds_dialogue_specific_messages(self, get_ffmpeg, run):
        run.return_value = SimpleNamespace(stdout=b"converted-mp3")
        session = Session(
            state="evaluate",
            evaluation_type=EVALUATION_DIALOGUE,
            prompt_text="Speaker 1: Hello. Speaker 2: Hi.",
            response_a=b"conversation-a",
            response_b=b"conversation-b",
        )

        messages = build_messages(session)
        content_text = " ".join(part.get("text", "") for part in messages[1]["content"])

        self.assertIn("Responsiveness to the Other Speaker", messages[0]["content"])
        self.assertIn("Multi-turn dialogue script", content_text)
        self.assertIn("Conversation A audio", content_text)
        self.assertIn("Conversation B audio", content_text)
        self.assertIn("conversation_a_classification", messages[0]["content"])

    def test_screenshot_prompt_uses_vision_model_for_exact_transcription(self):
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="Đây là prompt"))
        ]
        session = Session(state="evaluate", prompt_image=b"image-bytes", prompt_image_mime="image/jpeg")

        text = extract_prompt_from_image(client, session)

        self.assertEqual(text, "Đây là prompt")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "openai/gpt-4o-mini")
        self.assertTrue(request["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_report_escapes_generated_text_for_telegram_html(self):
        result = {
            "winner": "B",
            "dominant_factor": "Naturalness & clarity",
            "summary": "B wins (natural); A has <noise> and B sounds smoother.",
            "audio_recording_quality_verdict": "[Both Good]",
            "audio_recording_quality_breakdown": "A has light_noise; B is clean.",
            "persona_likeness_verdict": "[N/A]",
            "persona_likeness_breakdown": "No reference (none) was supplied.",
            "pronunciation_faithfulness_verdict": "[Both Good]",
            "pronunciation_breakdown": "A and B match the prompt.",
            "technical_string_verdict": "[N/A]",
            "naturalness_verdict": "[Response B]",
            "naturalness_breakdown": "A is stiff; B has natural pauses.",
        }

        report = format_result(result)

        self.assertIn("🎭 <b>Persona Likeness (Speaker Similarity): [N/A]</b>", report)
        self.assertIn("🔊 <b>Audio and Recording Quality: [Both Good]</b>", report)
        self.assertIn("🔢 <b>Technical String Check: [N/A]</b>", report)
        self.assertIn("\n\n<b>🏆 Final Verdict</b>\n", report)
        self.assertIn("\n\n<b>✍️ Justification (Ready to Copy-Paste)</b>", report)
        self.assertIn("<b>✍️ Justification (Ready to Copy-Paste)</b>", report)
        self.assertIn("A has light_noise", report)
        self.assertIn("B wins (natural); A has &lt;noise&gt;", report)
        self.assertIn("Naturalness &amp; clarity", report)

    def test_dialogue_report_uses_dialogue_verdicts_and_classifications(self):
        result = {
            "responsiveness_verdict": "[Conversation A]",
            "responsiveness_breakdown": "A reacts to jokes; B stays flat.",
            "conversational_naturalness_verdict": "[Conversation A]",
            "conversational_naturalness_breakdown": "A flows naturally; B sounds read.",
            "overall_preference": "[Conversation A]",
            "conversation_a_classification": "Human recording",
            "conversation_b_classification": "AI-generated",
            "justification": "Conversation A wins through engaged replies, while Conversation B sounds flatter and more scripted throughout the exchange.",
        }

        report = format_result(result, EVALUATION_DIALOGUE)

        self.assertIn("Multi-Speaker Evaluation Breakdown", report)
        self.assertIn("[Conversation A]", report)
        self.assertIn("Human recording", report)
        self.assertIn("AI-generated", report)
        self.assertNotIn("Persona likeness", report)


if __name__ == "__main__":
    unittest.main()