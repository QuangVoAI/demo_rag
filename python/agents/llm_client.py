"""
Groq LLM Client — Wrapper gọi Groq API cho Nhatrovn Assistant.

Tính năng:
  - Round-robin xoay vòng API key để tránh rate limit
  - Auto-truncation prompt khi vượt giới hạn token
  - Streaming completions (SSE)
  - Retry thông minh khi gặp lỗi 429/413
"""
import asyncio
import aiohttp
import json
import time
import tiktoken
from typing import AsyncGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, GROQ_API_KEYS

# Tích hợp Langfuse observability (bỏ qua nếu chưa cài)
try:
    from langfuse import observe as _observe
    def observe(**kwargs):
        """Wrapper giảm nhẹ nếu Langfuse chưa được cấu hình."""
        return _observe(**kwargs)
except ImportError:
    def observe(**kwargs):
        """No-op decorator khi không có langfuse."""
        def decorator(func):
            return func
        return decorator


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_SMART = "llama-3.3-70b-versatile"   # Mô hình thông minh, dùng cho sinh câu trả lời
GROQ_MODEL_FAST  = "llama-3.1-8b-instant"      # Mô hình nhanh, dùng cho routing / rewrite

# Giới hạn token đầu vào (Groq Free Tier)
GROQ_FREE_TIER_MAX_INPUT_TOKENS = 4500
_NUM_KEYS = max(1, len(GROQ_API_KEYS) if GROQ_API_KEYS else 1)
# Dãn khoảng cách giữa 2 lần gọi để tránh nghẽn TPM trên 70B
GROQ_RATE_LIMIT_DELAY = 6.0 / _NUM_KEYS

# Tokenizer tương thích OpenAI (GPT-4 / Llama dùng cùng cách đếm)
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Trạng thái round-robin key rotation
_groq_key_index = 0
_last_groq_call_time = 0.0
_rate_limit_lock = asyncio.Lock()
_BAD_GROQ_KEYS: set[str] = set()   # Các key đã bị block/hết hạn


# ---------------------------------------------------------------------------
# Tiện ích đếm và cắt token
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


def _truncate_text(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "\n[...đã cắt bớt...]"


def _truncate_messages(
    messages: list[dict],
    max_total_tokens: int = GROQ_FREE_TIER_MAX_INPUT_TOKENS,
) -> list[dict]:
    """Cắt ngắn messages để vừa giới hạn token của Groq Free Tier."""
    total = sum(_count_tokens(m.get("content", "")) for m in messages)
    if total <= max_total_tokens:
        return messages

    # Cắt message dài nhất để tiết kiệm nhất
    longest_idx = max(
        range(len(messages)),
        key=lambda i: _count_tokens(messages[i].get("content", "")),
    )
    overflow = total - max_total_tokens
    longest_tokens = _count_tokens(messages[longest_idx]["content"])
    new_max = max(500, longest_tokens - overflow)

    truncated = list(messages)
    truncated[longest_idx] = {
        **messages[longest_idx],
        "content": _truncate_text(messages[longest_idx]["content"], new_max),
    }
    return truncated


# ---------------------------------------------------------------------------
# Quản lý API key (round-robin, loại key lỗi)
# ---------------------------------------------------------------------------

def _get_groq_key() -> str:
    global _groq_key_index
    all_keys = GROQ_API_KEYS if GROQ_API_KEYS else [GROQ_API_KEY]
    valid_keys = [k for k in all_keys if k not in _BAD_GROQ_KEYS]
    if not valid_keys:
        raise RuntimeError("Tất cả Groq API key đã hết hạn hoặc bị giới hạn.")
    key = valid_keys[_groq_key_index % len(valid_keys)]
    _groq_key_index += 1
    return key


async def _respect_rate_limit() -> None:
    """Duy trì khoảng cách giữa các lần gọi để không vượt TPM."""
    global _last_groq_call_time
    async with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_groq_call_time
        if elapsed < GROQ_RATE_LIMIT_DELAY and _last_groq_call_time > 0:
            await asyncio.sleep(GROQ_RATE_LIMIT_DELAY - elapsed)
        _last_groq_call_time = time.time()


# ---------------------------------------------------------------------------
# Non-streaming completion
# ---------------------------------------------------------------------------

async def groq_complete(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Groq completion đơn giản (non-streaming)."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return await groq_chat_complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


@observe(name="groq_chat_complete", as_type="generation")
async def groq_chat_complete(
    messages: list[dict],
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Groq chat completion với auto-truncation và rate limiting."""
    api_key = _get_groq_key()
    await _respect_rate_limit()

    if model == GROQ_MODEL_FAST:
        messages = _truncate_messages(messages)
        max_tokens = min(max_tokens, 1500)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(5):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(25 * (attempt + 1))
                        api_key = _get_groq_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        continue

                    if resp.status == 413:
                        new_limit = int(GROQ_FREE_TIER_MAX_INPUT_TOKENS * (0.7 ** (attempt + 1)))
                        messages = _truncate_messages(messages, max_total_tokens=max(800, new_limit))
                        payload["messages"] = messages
                        await asyncio.sleep(25)
                        continue

                    if resp.status in {400, 401, 403}:
                        error_text = await resp.text()
                        if "restricted" in error_text.lower() or resp.status in {401, 403}:
                            _BAD_GROQ_KEYS.add(api_key)
                            api_key = _get_groq_key()
                            headers["Authorization"] = f"Bearer {api_key}"
                            continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Groq API lỗi ({resp.status}): {error_text[:300]}")

                    result = await resp.json()
                    choices = result.get("choices", [])
                    return choices[0].get("message", {}).get("content", "") if choices else ""

        except aiohttp.ClientError as e:
            if attempt == 4:
                raise RuntimeError(f"Groq connection lỗi sau 5 lần thử: {e}")
            await asyncio.sleep(5)

    return ""


# ---------------------------------------------------------------------------
# Streaming completion
# ---------------------------------------------------------------------------

async def groq_stream_complete(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """Groq streaming completion — yield từng token."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async for token in groq_stream_chat_complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        yield token


@observe(name="groq_stream_chat_complete", as_type="generation")
async def groq_stream_chat_complete(
    messages: list[dict],
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """Groq streaming chat completion — yield từng token chunk qua SSE."""
    api_key = _get_groq_key()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(25 * (attempt + 1))
                        api_key = _get_groq_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(f"Groq stream lỗi ({resp.status}): {error_text[:300]}")

                    # Đọc SSE stream và yield từng token
                    async for line in resp.content:
                        line_str = line.decode("utf-8").strip()
                        if not line_str or not line_str.startswith("data:"):
                            continue
                        data_str = line_str[5:].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data_str)
                            delta = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, IndexError):
                            continue
                    return

        except aiohttp.ClientError as e:
            if attempt == 2:
                raise RuntimeError(f"Groq stream lỗi sau 3 lần thử: {e}")
            await asyncio.sleep(5)
