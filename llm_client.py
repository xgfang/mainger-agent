"""
llm_client.py
-------------
Vendor-neutral wrapper covering five backends:

  - anthropic           : Claude family
  - openai              : GPT family (api.openai.com)
  - gemini              : Google Gemini family
  - openai_compatible   : any provider exposing the OpenAI Chat Completions API
                          (xAI Grok, Together, Fireworks, OpenRouter, Groq,
                          HuggingFace Inference Endpoints, vLLM, Ollama, etc.)

Adding a new closed-model vendor with native schema = new client class here.
Adding a new OpenAI-compatible provider = no code change; user supplies
`base_url` and a model name at request time.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[dict[str, Any]]   # [{"id":..., "name":..., "args":...}]
    stop_reason: str
    raw: Any


class LLMClient(Protocol):
    def complete(self, messages, tools, system) -> LLMResponse: ...
    def format_assistant_with_tools(self, resp: LLMResponse) -> dict: ...
    def format_tool_result(self, call_id: str, tool_name: str, result: dict) -> dict: ...


# --------------------------------------------------------------------------- #
# Anthropic (native)                                                           #
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, model, max_tokens, temperature, api_key=None, base_url=None):
        from anthropic import Anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not provided")
        kwargs = {"api_key": key}
        if base_url: kwargs["base_url"] = base_url
        self.client = Anthropic(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, messages, tools, system):
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            system=system, tools=tools, messages=messages,
        )
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "args": block.input})
        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls, stop_reason=resp.stop_reason, raw=resp,
        )

    def format_assistant_with_tools(self, resp):
        return {"role": "assistant", "content": resp.raw.content}

    def format_tool_result(self, call_id, tool_name, result):
        return {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }]}


# --------------------------------------------------------------------------- #
# OpenAI / OpenAI-compatible                                                   #
# --------------------------------------------------------------------------- #
class OpenAICompatibleClient:
    """Works with api.openai.com (default) and any OpenAI-compatible endpoint
    (xAI Grok, Together, Fireworks, OpenRouter, Groq, HF TGI, vLLM, Ollama)."""

    # Sensible default key env var per known base host
    _ENV_BY_HOST: dict[str, str] = {
        "api.openai.com":     "OPENAI_API_KEY",
        "api.x.ai":           "XAI_API_KEY",
        "api.together.xyz":   "TOGETHER_API_KEY",
        "api.fireworks.ai":   "FIREWORKS_API_KEY",
        "openrouter.ai":      "OPENROUTER_API_KEY",
        "api.groq.com":       "GROQ_API_KEY",
    }

    def __init__(self, model, max_tokens, temperature, api_key=None, base_url=None):
        from openai import OpenAI
        # Resolve API key from arg, then from env (host-specific or generic).
        key = api_key
        if not key and base_url:
            for host, env_name in self._ENV_BY_HOST.items():
                if host in base_url and os.environ.get(env_name):
                    key = os.environ[env_name]
                    break
        if not key:
            key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "API key not provided for OpenAI-compatible client. "
                "Pass api_key explicitly or set the relevant env var."
            )
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _to_openai_tools(tools):
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        } for t in tools]

    def complete(self, messages, tools, system):
        msgs = [{"role": "system", "content": system}] + messages
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            tools=self._to_openai_tools(tools), messages=msgs,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for c in msg.tool_calls:
                try:
                    args = json.loads(c.function.arguments) if c.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"id": c.id, "name": c.function.name, "args": args})
        return LLMResponse(
            text=msg.content, tool_calls=tool_calls,
            stop_reason=resp.choices[0].finish_reason, raw=msg,
        )

    def format_assistant_with_tools(self, resp):
        return resp.raw.model_dump(exclude_none=True)

    def format_tool_result(self, call_id, tool_name, result):
        return {
            "role": "tool", "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }


# --------------------------------------------------------------------------- #
# Google Gemini (native)                                                       #
# --------------------------------------------------------------------------- #
class GeminiClient:
    def __init__(self, model, max_tokens, temperature, api_key=None, base_url=None):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise RuntimeError(
                "Gemini support requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from e
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not provided")
        self.client = genai.Client(api_key=key)
        self.types = genai_types
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _to_gemini_tools(self, tools):
        # Gemini takes a list with a single Tool that contains many function declarations
        decls = [self.types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["input_schema"],
        ) for t in tools]
        return [self.types.Tool(function_declarations=decls)]

    @staticmethod
    def _to_gemini_messages(messages):
        """Translate our internal message log to Gemini `contents` format.

        Our log can contain assistant turns produced by Anthropic (with `content`
        as a list of blocks) or by OpenAI (with `content` string + `tool_calls`).
        For multi-turn with vendor switches, we emit a best-effort transcription.
        """
        contents = []
        for m in messages:
            role = m.get("role")
            if role == "user":
                # `content` may be a string or a list of blocks (e.g. tool_result)
                c = m.get("content")
                if isinstance(c, str):
                    contents.append({"role": "user", "parts": [{"text": c}]})
                elif isinstance(c, list):
                    parts = []
                    for blk in c:
                        if isinstance(blk, dict) and blk.get("type") == "tool_result":
                            parts.append({"function_response": {
                                "name": "tool",  # name is recoverable but not strictly required
                                "response": {"content": blk.get("content", "")},
                            }})
                        elif isinstance(blk, dict) and "text" in blk:
                            parts.append({"text": blk["text"]})
                    if parts:
                        contents.append({"role": "user", "parts": parts})
            elif role == "assistant":
                c = m.get("content")
                parts = []
                if isinstance(c, str):
                    parts.append({"text": c})
                elif isinstance(c, list):
                    for blk in c:
                        if hasattr(blk, "type"):
                            t = blk.type
                            if t == "text":
                                parts.append({"text": blk.text})
                            elif t == "tool_use":
                                parts.append({"function_call": {
                                    "name": blk.name,
                                    "args": blk.input,
                                }})
                if parts:
                    contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                contents.append({"role": "user", "parts": [{
                    "function_response": {
                        "name": m.get("name", "tool"),
                        "response": {"content": m.get("content", "")},
                    }
                }]})
        return contents

    def complete(self, messages, tools, system):
        cfg = self.types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            tools=self._to_gemini_tools(tools),
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=self._to_gemini_messages(messages),
            config=cfg,
        )

        text_parts, tool_calls = [], []
        # Gemini may put tool calls and text in candidates[0].content.parts
        try:
            parts = resp.candidates[0].content.parts or []
        except (IndexError, AttributeError):
            parts = []
        for part in parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                tool_calls.append({
                    "id": f"gemini_{fc.name}_{len(tool_calls)}",
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })

        finish = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
        stop = "tool_calls" if tool_calls else (str(finish) if finish else "stop")

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls, stop_reason=stop, raw=resp,
        )

    def format_assistant_with_tools(self, resp):
        # Mirror Anthropic's shape so the rest of the pipeline doesn't care
        content_blocks = []
        try:
            parts = resp.raw.candidates[0].content.parts or []
        except (IndexError, AttributeError):
            parts = []
        for part in parts:
            if getattr(part, "text", None):
                content_blocks.append({"type": "text", "text": part.text})
            fc = getattr(part, "function_call", None)
            if fc is not None:
                content_blocks.append({
                    "type": "tool_use",
                    "id": f"gemini_{fc.name}",
                    "name": fc.name,
                    "input": dict(fc.args) if fc.args else {},
                })
        return {"role": "assistant", "content": content_blocks}

    def format_tool_result(self, call_id, tool_name, result):
        return {
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        }


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def make_client(
    vendor: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Create a vendor client.

    `vendor` is one of:
        anthropic | openai | gemini | xai | grok | openai_compatible | together |
        fireworks | openrouter | groq | huggingface | custom

    For the "openai_compatible-family" vendors, `base_url` is auto-populated
    from a known map below if not supplied. For 'custom', `base_url` is
    required.
    """
    vendor = (vendor or "").lower()

    # Map convenience vendor names to (real_vendor, default_base_url)
    OAI_COMPAT_PRESETS = {
        "xai":           ("openai_compatible", "https://api.x.ai/v1"),
        "grok":          ("openai_compatible", "https://api.x.ai/v1"),
        "together":      ("openai_compatible", "https://api.together.xyz/v1"),
        "fireworks":     ("openai_compatible", "https://api.fireworks.ai/inference/v1"),
        "openrouter":    ("openai_compatible", "https://openrouter.ai/api/v1"),
        "groq":          ("openai_compatible", "https://api.groq.com/openai/v1"),
        "huggingface":   ("openai_compatible", "https://router.huggingface.co/v1"),
    }

    if vendor in OAI_COMPAT_PRESETS:
        real, default_url = OAI_COMPAT_PRESETS[vendor]
        return OpenAICompatibleClient(
            model, max_tokens, temperature,
            api_key=api_key, base_url=base_url or default_url,
        )

    if vendor == "anthropic":
        return AnthropicClient(model, max_tokens, temperature,
                               api_key=api_key, base_url=base_url)
    if vendor == "openai":
        return OpenAICompatibleClient(model, max_tokens, temperature,
                                      api_key=api_key, base_url=base_url)
    if vendor == "gemini":
        return GeminiClient(model, max_tokens, temperature,
                            api_key=api_key, base_url=base_url)
    if vendor in ("openai_compatible", "custom"):
        if not base_url:
            raise ValueError(
                f"vendor='{vendor}' requires a base_url "
                "(e.g. https://api.together.xyz/v1)"
            )
        return OpenAICompatibleClient(model, max_tokens, temperature,
                                      api_key=api_key, base_url=base_url)

    raise ValueError(
        f"Unknown vendor '{vendor}'. Supported: anthropic, openai, gemini, "
        "xai, together, fireworks, openrouter, groq, huggingface, custom."
    )
