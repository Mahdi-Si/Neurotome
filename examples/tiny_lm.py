"""A tiny causal LM with a character tokenizer: `py:examples.tiny_lm:build` (offline; tests and e2e).

Run from the repo root (needs the hf extra):

    neurotome capture py:examples.tiny_lm:build --text "the cat sat on the mat"

A module may define TOKENIZER (encode(text) -> ids, decode(ids) -> str, pad_token_id / eos_token_id):
Neurotome then accepts text samples, labels tokens and offers the logit lens.
"""

import torch

VOCAB = "\0 abcdefghijklmnopqrstuvwxyz.,!?'"  # "\0" is the pad / eos token (id 0)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [VOCAB.find(c) if c in VOCAB[1:] else VOCAB.index("?") for c in text.lower()]

    def decode(self, ids: list[int]) -> str:
        return "".join(VOCAB[i] for i in ids if 0 < i < len(VOCAB))


TOKENIZER = CharTokenizer()


def build():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_positions=128, vocab_size=len(VOCAB),
                     bos_token_id=0, eos_token_id=0)  # fmt: skip
    model = GPT2LMHeadModel(cfg)
    model.set_attn_implementation("eager")
    model.config.use_cache = False
    return model
