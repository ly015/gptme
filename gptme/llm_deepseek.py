import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

from .config import Config
from .constants import TEMPERATURE, TOP_P
from .message import Message, msgs2dicts

if TYPE_CHECKING:
    from openai import OpenAI

deepseek: "DeepSeek | None" = None
logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def init(llm: str, config: Config):
    global deepseek  # noqa
    from openai import OpenAI  # fmt

    api_key = config.get_env_required("DEEPSEEK_API_KEY")
    deepseek = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    assert deepseek, "LLM not initialized"


def get_client() -> "DeepSeek | None":
    return deepseek


def chat(messages: list[Message], model: str) -> str:
    assert deepseek, "LLM not initialized"
    # noreorder

    response = deepseek.chat.completions.create(
        model=model,
        messages=msgs2dicts(messages, openai=False),  # type: ignore
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    content = response.choices[0].message.content
    assert content
    return content


def stream(messages: list[Message], model: str) -> Generator[str, None, None]:
    assert deepseek, "LLM not initialized"
    stop_reason = None

    for chunk in deepseek.chat.completions.create(
        model=model,
        messages=msgs2dicts(messages, openai=False),  # type: ignore
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
    ):
        if not chunk.choices:  # type: ignore
            # Got a chunk with no choices, Azure always sends one of these at the start
            continue
        stop_reason = chunk.choices[0].finish_reason  # type: ignore
        content = chunk.choices[0].delta.content  # type: ignore
        if content:
            yield content
    logger.debug(f"Stop reason: {stop_reason}")
