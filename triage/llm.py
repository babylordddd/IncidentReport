"""Approach 2 - zero-shot LLM classifier with structured (JSON-schema) output.

Provider is chosen by the model name:
  gemini-*            Google Gemini API (free tier via AI Studio key in GEMINI_API_KEY)
  ollama/<model>      local model served by Ollama at http://localhost:11434 (no key, no cost)
  claude-*            Anthropic API (ANTHROPIC_API_KEY), kept as an optional paid comparison
"""

import json
import os
import random
import time
import urllib.request
from typing import Literal

from pydantic import create_model

from .taxonomy import CATEGORIES_V1, CATEGORY_DESCRIPTIONS, URGENCY_DEFINITION

DEFAULT_MODEL = os.environ.get("TRIAGE_LLM", "gemini-3.5-flash-lite")
LOCAL_MODEL = "ollama/qwen2.5:7b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# USD per 1M tokens (input, output) at PAID list prices (ai.google.dev pricing, checked 2026-10-27),
# used to cost the system at production volume: free tiers are rate-limited and not a production
# option. Ollama runs locally: $0 API cost.
PRICES = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def model_slug(model):
    """Filesystem-safe model name for cache files."""
    return model.replace("/", "_").replace(":", "-")


def system_prompt(categories):
    cats = "\n".join(f"- {c}: {CATEGORY_DESCRIPTIONS[c]}" for c in categories)
    return f"""You triage incident reports for a logistics operator (trucking, warehouses, ports).

Classify each report into exactly one category:
{cats}

Precedence: if anyone is hurt or hazardous material is released, the category is safety_incident.

Urgency: {URGENCY_DEFINITION}

Also give:
- urgency_probability: your probability (0-1) that the case is urgent.
- category_confidence: your probability (0-1) that the category is correct.
- rationale: one short sentence.
Judge urgency by the operational facts, not by the reporter's tone, capital letters or the word "urgent".
Answer with JSON only."""


def output_schema(categories):
    return create_model(
        "Triage",
        category=(Literal[tuple(categories)], ...),
        urgency=(Literal["urgent", "normal"], ...),
        urgency_probability=(float, ...),
        category_confidence=(float, ...),
        rationale=(str, ...),
    )


class LLMTriage:
    def __init__(self, model=DEFAULT_MODEL, categories=CATEGORIES_V1):
        self.model = model
        self.categories = list(categories)
        self.system = system_prompt(self.categories)
        self.schema = output_schema(self.categories)
        if model.startswith("gemini"):
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                       http_options=types.HttpOptions(timeout=60_000))
            self._call = self._gemini
        elif model.startswith("ollama/"):
            self._call = self._ollama
        elif model.startswith("claude"):
            import anthropic
            self.client = anthropic.Anthropic()
            self._call = self._anthropic
        else:
            raise ValueError(f"unknown model provider: {model}")

    # Each backend returns (parsed dict or None, input_tokens, output_tokens, stop_info).

    def _gemini(self, text):
        from google.genai import errors, types
        config = types.GenerateContentConfig(
            system_instruction=self.system,
            response_mime_type="application/json",
            response_schema=self.schema,
            temperature=0,
            # Classification needs little reasoning: keep thinking (and cost/latency) minimal.
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=text, config=config)
                break
            except errors.APIError as e:
                if e.code in (429, 500, 503, 504) and attempt < 5:  # free-tier rate limit / overload
                    time.sleep(min(60, 4 * 2 ** attempt) + random.random())
                    continue
                raise
        u = resp.usage_metadata
        out_tokens = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        parsed = resp.parsed.model_dump() if resp.parsed is not None else None
        return parsed, u.prompt_token_count or 0, out_tokens, None

    def _ollama(self, text):
        body = json.dumps({
            "model": self.model.split("/", 1)[1],
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
            "format": self.schema.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
        try:
            parsed = self.schema.model_validate_json(data["message"]["content"]).model_dump()
        except ValueError:
            parsed = None
        return parsed, data.get("prompt_eval_count", 0), data.get("eval_count", 0), None

    def _anthropic(self, text):
        resp = self.client.messages.parse(
            model=self.model, max_tokens=4000, system=self.system,
            output_config={"effort": "low"}, output_format=self.schema,
            messages=[{"role": "user", "content": text}],
        )
        parsed = resp.parsed_output.model_dump() if resp.parsed_output is not None else None
        return parsed, resp.usage.input_tokens, resp.usage.output_tokens, resp.stop_reason

    def predict(self, text):
        t0 = time.perf_counter()
        parsed, tin, tout, stop = self._call(f"<report>\n{text}\n</report>")
        latency_ms = (time.perf_counter() - t0) * 1000
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        base = {
            "latency_ms": latency_ms,
            "cost_usd": (tin * pin + tout * pout) / 1e6,
            "input_tokens": tin,
            "output_tokens": tout,
            "model": self.model,
        }
        if parsed is None:
            # A refused or unparseable answer is sent to a human, never guessed.
            return {**base, "category": None, "urgency": None, "p_urgent": 0.5, "category_confidence": 0.0,
                    "category_probs": {}, "rationale": f"no valid output (stop={stop})", "error": True}
        conf = min(max(parsed["category_confidence"], 0.0), 1.0)
        return {
            **base,
            "category": parsed["category"],
            "urgency": parsed["urgency"],
            "p_urgent": min(max(parsed["urgency_probability"], 0.0), 1.0),
            "category_confidence": conf,
            "category_probs": {parsed["category"]: conf},
            "rationale": parsed["rationale"],
        }
