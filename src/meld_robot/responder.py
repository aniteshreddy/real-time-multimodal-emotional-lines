"""Response generation: Qwen2.5-1.5B-Instruct (~1.54B params) prompted with
the structured multimodal state.

The LLM never sees raw audio or pixels: it consumes the perception stack's
structured output (emotion tag + evidence), which is exactly the challenge
requirement "response text grounded in the multimodal input". Its parameters
count toward the 6B budget (see notebook + README tables).
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from . import config

SYSTEM_PROMPT = (
    "You are the emotive core of a companion robot talking with one person. "
    "You receive STRUCTURED STATE from the robot's multimodal perception stack "
    "(emotion classification plus evidence from text, voice and face). "
    "First emit one <evidence> line summarizing which signals you used - cite at "
    "least two different modalities. Then emit <response> with a reply to the "
    "person: warm, specific to their state, at most two short sentences. "
    "Only use the signals provided; never invent facts about the person. "
    "Example:\n"
    "<evidence> flat voice with low pitch, no smile, text said everything is fine\n"
    "<response> You did not sound fine just now. What is going on?"
)


class PromptedResponder:
    def __init__(self, model_id: str = config.RESPONDER_ID,
                 device: torch.device | None = None):
        from .utils import get_device
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device).eval()

    # ------------------------------------------------------------------
    def build_messages(self, state: dict, context: str | None = None) -> list[dict]:
        ev = state["evidence"]
        top3 = ", ".join(f"{e}({p})" for e, p in state["top3"])
        user = (
            f"STRUCTURED STATE for the user's last utterance:\n"
            f"- dominant emotion: {state['emotion']} (confidence {state['confidence']})\n"
            f"- runner-up emotions: {top3}\n"
            f"- text heard: {ev['text']}\n"
            f"- voice evidence: {'; '.join(ev['audio'])}\n"
            f"- face evidence: {'; '.join(ev['face'])}\n"
        )
        if context:
            user += f"- conversation context (previous turn): {context}\n"
        user += "\nProduce the <evidence> line then the <response>."
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        return messages

    # ------------------------------------------------------------------
    def generate(self, state: dict, context: str | None = None) -> tuple[str, dict]:
        """Non-streaming generation; returns (text, timing)."""
        messages = self.build_messages(state, context)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.RESPONSE_TEMPERATURE, do_sample=True,
                top_p=0.9, pad_token_id=self.tokenizer.eos_token_id,
            )
        timing = {"generation_ms": (time.perf_counter() - t0) * 1000,
                  "new_tokens": int(out.shape[1] - inputs["input_ids"].shape[1])}
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, timing

    def generate_stream(self, state: dict, context: str | None = None):
        """Yield reply tokens as they are produced (streaming text output)."""
        messages = self.build_messages(state, context)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs, max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.RESPONSE_TEMPERATURE, do_sample=True,
            top_p=0.9, pad_token_id=self.tokenizer.eos_token_id, streamer=streamer,
        )
        from threading import Thread
        thread = Thread(target=self._generate_thread, kwargs={"gen_kwargs": gen_kwargs})
        thread.start()
        for token in streamer:
            yield token
        thread.join()

    def _generate_thread(self, gen_kwargs):
        with torch.no_grad():
            self.model.generate(**gen_kwargs)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
