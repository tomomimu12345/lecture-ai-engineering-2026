from types import SimpleNamespace
import sys

from app import ask as ask_module


def test_chat_sends_follow_up_messages_to_same_session(monkeypatch, capsys):
    messages = []
    configs = []

    class FakeConversation:
        def send_message(self, message, config=None):
            messages.append(message)
            return SimpleNamespace(text=f"response {len(messages)}", usage_metadata=None)

    class FakeChats:
        def create(self, *, model, config):
            assert model == ask_module.MODEL
            configs.append(config)
            return conversation

    conversation = FakeConversation()
    client = SimpleNamespace(chats=FakeChats())
    prompts = iter(["first question", "follow-up", "/exit"])
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    ask_module.chat(temperature=0.4)

    assert messages == ["first question", "follow-up"]
    assert len(configs) == 1
    assert configs[0].temperature == 0.4
    assert "response 2" in capsys.readouterr().out


def test_chat_accepts_initial_question_and_temperature(monkeypatch):
    arguments = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.ask", "LoRAとは何ですか？", "--chat", "--temperature", "0.4"],
    )
    monkeypatch.setattr(
        ask_module,
        "chat",
        lambda question=None, temperature=None: arguments.update(
            question=question, temperature=temperature
        ),
    )

    ask_module.main()

    assert arguments == {"question": "LoRAとは何ですか？", "temperature": 0.4}


def test_chat_sends_initial_question_before_follow_up(monkeypatch):
    messages = []

    class FakeConversation:
        def send_message(self, message, config=None):
            messages.append(message)
            return SimpleNamespace(text="response", usage_metadata=None)

    conversation = FakeConversation()
    client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: conversation)
    )
    prompts = iter(["follow-up", "/exit"])
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    ask_module.chat(question="initial question")

    assert messages == ["initial question", "follow-up"]