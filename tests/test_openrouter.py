import base64
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bot import Session, audio_input_part, build_messages, extract_prompt_from_image, format_result


class OpenRouterPayloadTests(unittest.TestCase):
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

        self.assertIn("<b>Persona likeness (Speaker Similarity):</b>", report)
        self.assertIn("<b>✍️ Justification (Ready to Copy-Paste)</b>", report)
        self.assertIn("A has light_noise", report)
        self.assertIn("B wins (natural); A has &lt;noise&gt;", report)
        self.assertIn("Naturalness &amp; clarity", report)


if __name__ == "__main__":
    unittest.main()