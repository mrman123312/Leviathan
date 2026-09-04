from __future__ import annotations

import unittest

from leviathan.inference_client import (
    ChatRequest,
    CompletionRequest,
    extract_chat_text,
    extract_completion_text,
    normalize_base_url,
)


class InferenceClientTests(unittest.TestCase):
    def test_completion_payload_targets_base_model_prompting(self) -> None:
        payload = CompletionRequest(
            model="deepseek-ai/DeepSeek-V4-Pro-Base",
            prompt="The capital of France is",
            max_tokens=16,
            temperature=0.0,
        ).as_dict()
        self.assertEqual(payload["model"], "deepseek-ai/DeepSeek-V4-Pro-Base")
        self.assertEqual(payload["prompt"], "The capital of France is")
        self.assertEqual(payload["max_tokens"], 16)
        self.assertEqual(payload["temperature"], 0.0)

    def test_chat_payload_is_available_for_posttrained_or_templated_servers(self) -> None:
        payload = ChatRequest(
            model="served-v4",
            messages=[{"role": "user", "content": "hello"}],
        ).as_dict()
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "hello")

    def test_completion_response_extraction(self) -> None:
        text = extract_completion_text({"choices": [{"text": " Paris."}]})
        self.assertEqual(text, " Paris.")

    def test_chat_response_extraction(self) -> None:
        text = extract_chat_text(
            {"choices": [{"message": {"role": "assistant", "content": "Paris"}}]}
        )
        self.assertEqual(text, "Paris")

    def test_malformed_responses_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            extract_completion_text({"choices": []})
        with self.assertRaises(ValueError):
            extract_chat_text({"choices": [{"message": {}}]})

    def test_base_url_normalization(self) -> None:
        self.assertEqual(normalize_base_url("http://127.0.0.1:8000/"), "http://127.0.0.1:8000")


if __name__ == "__main__":
    unittest.main()
