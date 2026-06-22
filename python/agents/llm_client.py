"""
Groq LLM Client — Wrapper cho Groq API (stream + non-stream).

Extracted từ lightrag_setup/llm_wrapper.py và mở rộng thêm streaming.
Hỗ trợ:
  - Round-robin API key rotation
  - Rate limiting cho Free Tier
  - Auto-truncation prompt
  - Streaming completions (SSE)
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
from config import (
    GROQ_API_KEY, GROQ_API_KEYS,
    VERTEX_PROJECT_ID, VERTEX_REGION
)

# Langfuse decorator (graceful fallback if not configured)
try:
    from langfuse import observe as _observe
    def observe(**kwargs):
        """Wrapper that silently degrades if Langfuse is not configured."""
        return _observe(**kwargs)
except ImportError:
    def observe(**kwargs):
        """No-op decorator when langfuse not installed."""
        def decorator(func):
            return func
        return decorator


# ─── Constants ───────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_SMART = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"

# Token limits
GROQ_FREE_TIER_MAX_INPUT_TOKENS = 4500
_NUM_KEYS = max(1, len(GROQ_API_KEYS) if GROQ_API_KEYS else 1)
GROQ_RATE_LIMIT_DELAY = 6.0 / _NUM_KEYS  # Dãn cách rộng hơn để Groq 70B không bị nghẽn TPM

# Tokenizer
_ENCODER = tiktoken.get_encoding("cl100k_base")

# Round-robin state
_groq_key_index = 0
_last_groq_call_time = 0.0
_rate_limit_lock = asyncio.Lock()
_BAD_GROQ_KEYS = set()


# ─── Token Utilities ─────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


def _truncate_text(text: str, max_tokens: int) -> str:
    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]) + "\n[...truncated...]"


def _truncate_messages(
    messages: list[dict],
    max_total_tokens: int = GROQ_FREE_TIER_MAX_INPUT_TOKENS,
) -> list[dict]:
    """Cắt ngắn messages để vừa giới hạn TPM."""
    total = sum(_count_tokens(m.get("content", "")) for m in messages)
    if total <= max_total_tokens:
        return messages

    longest_idx = max(
        range(len(messages)),
        key=lambda i: _count_tokens(messages[i].get("content", "")),
    )

    overflow = total - max_total_tokens
    longest_content = messages[longest_idx]["content"]
    longest_tokens = _count_tokens(longest_content)
    new_max = max(500, longest_tokens - overflow)

    truncated = list(messages)
    truncated[longest_idx] = {
        **messages[longest_idx],
        "content": _truncate_text(longest_content, new_max),
    }
    return truncated


# ─── Key Management ──────────────────────────────────────

def _get_groq_key() -> str:
    global _groq_key_index
    all_keys = GROQ_API_KEYS if GROQ_API_KEYS else [GROQ_API_KEY]
    
    # Filter valid keys
    valid_keys = [k for k in all_keys if k not in _BAD_GROQ_KEYS]
    if not valid_keys:
        raise RuntimeError("All Groq API keys are invalid or restricted.")
    
    key = valid_keys[_groq_key_index % len(valid_keys)]
    _groq_key_index += 1
    return key


async def _respect_rate_limit():
    global _last_groq_call_time
    async with _rate_limit_lock:
        now = time.time()
        elapsed = now - _last_groq_call_time
        if elapsed < GROQ_RATE_LIMIT_DELAY and _last_groq_call_time > 0:
            wait = GROQ_RATE_LIMIT_DELAY - elapsed
            await asyncio.sleep(wait)
        _last_groq_call_time = time.time()


# ─── Vertex AI Gemini Completion ─────────────────────────────

@observe(name="vertex_gemini_complete", as_type="generation")
async def vertex_gemini_complete(
    messages: list[dict],
    model: str = "gemini-1.5-flash",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Vertex AI Gemini chat completion using Google Cloud SDK."""
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content, Part
    
    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
    
    # Extract system instruction and format contents
    system_instruction = None
    contents = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            system_instruction = content
        else:
            # Vertex roles: 'user', 'model'
            v_role = "user" if role != "assistant" else "model"
            contents.append(Content(role=v_role, parts=[Part.from_text(content)]))
            
    v_model = GenerativeModel(model, system_instruction=system_instruction)
    
    response = await v_model.generate_content_async(
        contents,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    )
    return response.text


# ─── Vertex AI Custom Endpoint (Fine-tuned Model) ─────────────────────────────

# Cache endpoint URL and access token
_vertex_endpoint_url: str | None = None
_vertex_access_token: str = ""
_vertex_token_expiry: float = 0.0
_VERTEX_TOKEN_TTL = 3000  # Refresh token every 50 minutes (GCP tokens last ~60min)


def _get_vertex_endpoint_url() -> str:
    """Build Vertex AI Custom Endpoint URL from project config."""
    global _vertex_endpoint_url
    if _vertex_endpoint_url is not None:
        return _vertex_endpoint_url
    
    # Format: https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/REGION/endpoints/ENDPOINT_ID
    # Endpoint ID can be set via env or inferred
    import os
    endpoint_id = os.getenv("VERTEX_ENDPOINT_ID", "")
    
    if endpoint_id:
        _vertex_endpoint_url = (
            f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/"
            f"projects/{VERTEX_PROJECT_ID}/locations/{VERTEX_REGION}/"
            f"endpoints/{endpoint_id}"
        )
    else:
        # Try to get from gcloud or raise error
        _vertex_endpoint_url = os.getenv("VERTEX_ENDPOINT_URL", "")
    
    return _vertex_endpoint_url


@observe(name="vertex_custom_complete", as_type="generation")
async def vertex_custom_complete(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Vertex AI Custom Endpoint completion (fine-tuned Llama 3.1 8B).
    Uses application/json format compatible with our serve.py container.
    """
    global _vertex_access_token, _vertex_token_expiry
    import os
    
    endpoint_url = _get_vertex_endpoint_url()
    if not endpoint_url:
        raise RuntimeError(
            "Vertex AI Endpoint not configured. Set VERTEX_ENDPOINT_ID or VERTEX_ENDPOINT_URL"
        )
    
    # Get access token — cached for ~50 minutes
    now = time.time()
    if _vertex_access_token and now < _vertex_token_expiry:
        access_token = _vertex_access_token
    else:
        access_token = os.getenv("VERTEX_ACCESS_TOKEN", "")
        if not access_token:
            try:
                # Try google-auth library first (faster, no subprocess)
                import google.auth
                import google.auth.transport.requests
                credentials, _ = google.auth.default()
                credentials.refresh(google.auth.transport.requests.Request())
                access_token = credentials.token
            except Exception:
                # Fallback to gcloud CLI
                try:
                    import subprocess
                    result = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        capture_output=True, text=True, timeout=30
                    )
                    if result.returncode == 0:
                        access_token = result.stdout.strip()
                except Exception:
                    pass
        
        if not access_token:
            raise RuntimeError(
                "GCP access token not available. Run 'gcloud auth application-default login' "
                "or set VERTEX_ACCESS_TOKEN environment variable."
            )
        # Cache token
        _vertex_access_token = access_token
        _vertex_token_expiry = now + _VERTEX_TOKEN_TTL
    
    # Vertex AI rawPredict accepts the direct JSON payload
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{endpoint_url}:rawPredict",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(
                    f"Vertex AI Endpoint error ({resp.status}): {error[:500]}"
                )
            
            result = await resp.json()
            
            # Since we use :rawPredict, result is the direct response from the container
            if "reply" in result:
                return result["reply"]
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0].get("message", {}).get("content", "")

            # Fallback if we switch back to :predict wrapper in the future
            predictions = result.get("predictions", [])
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                if isinstance(pred, dict):
                    if "reply" in pred:
                        return pred["reply"]
                    if "choices" in pred:
                        return pred["choices"][0].get("message", {}).get("content", "")
                return str(pred)
            
            return ""


# Alias for backward compatibility - vertex mode uses custom endpoint for fine-tuned model
async def vertex_chat_complete(
    messages: list[dict],
    model: str = "gemini-1.5-flash",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """
    Vertex AI chat completion - auto-selects Gemini or Custom Endpoint.
    If model starts with 'projects/' it's a custom endpoint, else Gemini.
    """
    if model.startswith("projects/") or model.startswith("custom:"):
        return await vertex_custom_complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return await vertex_gemini_complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ─── Non-Streaming Completion ────────────────────────────

async def groq_complete(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Groq completion (non-streaming)."""
    messages = []
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

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=900),
                ) as resp:
                    if resp.status == 429:
                        wait_time = 25 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        api_key = _get_groq_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        continue

                    if resp.status == 413:
                        new_limit = int(
                            GROQ_FREE_TIER_MAX_INPUT_TOKENS
                            * (0.7 ** (attempt + 1))
                        )
                        messages = _truncate_messages(
                            messages, max_total_tokens=max(800, new_limit)
                        )
                        payload["messages"] = messages
                        await asyncio.sleep(25)
                        continue

                    if resp.status in [400, 401, 403]:
                        error_text = await resp.text()
                        if "restricted" in error_text.lower() or resp.status in [401, 403]:
                            _BAD_GROQ_KEYS.add(api_key)
                            api_key = _get_groq_key()
                            headers["Authorization"] = f"Bearer {api_key}"
                            continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Groq API error ({resp.status}): {error_text[:300]}"
                        )

                    result = await resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    return ""
        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Groq connection error after {max_retries} retries: {e}"
                )
            await asyncio.sleep(5)

    return ""


# ─── Streaming Completion ────────────────────────────────

async def groq_stream_complete(
    prompt: str,
    system_prompt: str = "",
    model: str = GROQ_MODEL_SMART,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> AsyncGenerator[str, None]:
    """
    Groq streaming completion — yield từng token.
    Sử dụng SSE (Server-Sent Events) từ Groq API.
    """
    messages = []
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
    """Groq streaming chat completion — yield từng token chunk."""
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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROQ_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    if resp.status == 429:
                        wait_time = 25 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        api_key = _get_groq_key()
                        headers["Authorization"] = f"Bearer {api_key}"
                        continue

                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Groq stream error ({resp.status}): {error_text[:300]}"
                        )

                    # Parse SSE stream
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

                    return  # Stream completed
        except aiohttp.ClientError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Groq stream error after retries: {e}")
            await asyncio.sleep(5)


