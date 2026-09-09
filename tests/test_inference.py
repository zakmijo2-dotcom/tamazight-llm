"""
Tests for inference functions and formatting.
"""

from src.chat_template import build_messages

def test_inference_message_construction():
    msgs = build_messages(
        user_text="Azul!",
        system_prompt="System prompt.",
        history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    )
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["role"] == "user"
    assert msgs[3]["content"] == "Azul!"
