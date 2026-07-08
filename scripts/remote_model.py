"""Drop-in stand-in for unsloth.FastModel that delegates generation to vLLM.

Scripts in this repo are written against the unsloth API:

    model, tokenizer = FastModel.from_pretrained(name_or_adapter_dir, ...)
    FastModel.for_inference(model)
    inputs = tokenizer(text=prompt, return_tensors="pt", ...).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=..., do_sample=False)
    tokenizer.decode(output_ids[0][n:], skip_special_tokens=True)

This module implements just enough of that surface to run the same scripts on
a machine with no GPU, no CUDA, and no torch:

  * The tokenizer is the real Hugging Face tokenizer/processor, loaded locally
    (CPU-only). Chat templating and tokenization are therefore identical to
    the in-process path.
  * tokenizer(...) returns numpy arrays instead of torch tensors (so torch is
    not required); .to(device) is a no-op.
  * model.generate() POSTs the prompt *token ids* to a vLLM OpenAI-compatible
    server (/v1/completions accepts token-id prompts), then re-encodes the
    completion so the caller's decode-by-slicing still works.

Configuration (env vars):
    VLLM_BASE_URL  server base URL, default http://localhost:8000
    VLLM_MODEL     served model name to request. Defaults to the HF model id,
                   or to the directory basename when given a local adapter dir
                   (which must match the name in the server's --lora-modules).
"""

import json
import os
from pathlib import Path

import numpy as np
import requests


class _Batch(dict):
    """Stands in for transformers.BatchEncoding: a mapping with a no-op .to()."""

    def to(self, device):
        return self


class _RemoteTokenizer:
    """Wraps the real HF tokenizer/processor; only __call__ is intercepted."""

    def __init__(self, hf_tokenizer):
        self._hf = hf_tokenizer

    def __getattr__(self, name):
        # apply_chat_template, decode, etc. hit the real tokenizer.
        return getattr(self._hf, name)

    def __call__(self, *args, return_tensors=None, **kwargs):
        # Force numpy so torch is not needed locally.
        enc = self._hf(*args, return_tensors="np", **kwargs)
        return _Batch(dict(enc))


class _RemoteModel:
    device = "cpu"  # only ever passed to _Batch.to(), which ignores it

    def __init__(self, served_name, hf_tokenizer, base_url):
        self._served = served_name
        self._hf = hf_tokenizer
        self._url = base_url.rstrip("/")

    def generate(self, input_ids=None, max_new_tokens=128, do_sample=False, **_ignored):
        prompt_ids = np.asarray(input_ids)[0].tolist()
        resp = requests.post(
            f"{self._url}/v1/completions",
            json={
                "model": self._served,
                "prompt": prompt_ids,
                "max_tokens": int(max_new_tokens),
                "temperature": 1.0 if do_sample else 0.0,
            },
            timeout=300,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["text"]
        completion_ids = (
            self._hf(text=text, add_special_tokens=False, return_tensors="np")
            ["input_ids"][0]
            .tolist()
        )
        return np.asarray([prompt_ids + completion_ids])


def _check_server(base_url, served_name):
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/v1/models", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot reach vLLM at {base_url} — is the server running and the "
            "port forwarded? (kubectl port-forward svc/vllm 8000:8000)"
        ) from e
    names = [m["id"] for m in resp.json().get("data", [])]
    if served_name not in names:
        raise RuntimeError(
            f"vLLM at {base_url} does not serve {served_name!r}; it serves {names}. "
            "Set VLLM_MODEL or check the server's --lora-modules names."
        )


class FastModel:
    @staticmethod
    def from_pretrained(model_name, **_ignored):
        from transformers import AutoProcessor, AutoTokenizer

        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        path = Path(model_name)
        tokenizer_source = model_name
        if path.is_dir():
            served_name = os.environ.get("VLLM_MODEL", path.name)
            if not (path / "tokenizer_config.json").exists():
                # Adapter dir without tokenizer files: load it from the base model.
                cfg = json.loads((path / "adapter_config.json").read_text())
                tokenizer_source = cfg["base_model_name_or_path"]
        else:
            served_name = os.environ.get("VLLM_MODEL", model_name)

        _check_server(base_url, served_name)

        try:
            hf_tokenizer = AutoProcessor.from_pretrained(tokenizer_source)
        except Exception:
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        model = _RemoteModel(served_name, hf_tokenizer, base_url)
        return model, _RemoteTokenizer(hf_tokenizer)

    @staticmethod
    def for_inference(model):
        return model
