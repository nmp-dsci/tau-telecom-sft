"""The trainer's pieces that need no model download: block-wise attention and the mask check."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tau_telecom_sft.train import attention  # noqa: E402
from tau_telecom_sft.train.mask import check, segments  # noqa: E402


def test_chunked_attention_matches_sdpa_with_gradients() -> None:
    from torch.nn import functional

    torch.manual_seed(0)
    old = attention.BLOCK
    attention.BLOCK = 64
    try:
        q = torch.randn(1, 4, 200, 16, requires_grad=True)
        k = torch.randn(1, 2, 200, 16, requires_grad=True)
        v = torch.randn(1, 2, 200, 16, requires_grad=True)
        out, _ = attention.chunked_sdpa(None, q, k, v, None, scaling=0.25)
        ref = functional.scaled_dot_product_attention(
            q, attention._repeat_kv(k, 2), attention._repeat_kv(v, 2), is_causal=True, scale=0.25
        ).transpose(1, 2)
        assert torch.allclose(out, ref, atol=1e-5)
        g1 = torch.autograd.grad(out.square().sum(), (q, k, v))
        g2 = torch.autograd.grad(ref.square().sum(), (q, k, v))
        assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(g1, g2, strict=True))
    finally:
        attention.BLOCK = old


class _Tok:
    """A tokenizer whose ids are characters, so segments decode to the exact text."""

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids)


def _render(row: dict, trained_turns: set[int]) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []

    def put(text: str, m: int) -> None:
        ids.extend(ord(c) for c in text)
        mask.extend([m] * len(text))

    k = 0
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            put("<|im_start|>assistant\n", 0)
            body = msg["content"]
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                body += f'\n<tool_call>\n{{"name": "{fn["name"]}", "arguments": {fn["arguments"]}}}\n</tool_call>'
            put(body + "<|im_end|>\n", int(k in trained_turns))
            k += 1
        else:
            put(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n", 0)
    return ids, mask


ROW = {
    "messages": [
        {"role": "system", "content": "You are a telecom agent. Follow the policy below."},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "My mobile data stopped working while I am abroad."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "get_customer_by_phone",
                        "arguments": '{"phone_number": "555-123-2002"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"customer_id": "C1001", "line": "L1"}'},
        {"role": "assistant", "content": "Roaming is off on your line. Shall I turn it on?"},
    ]
}


def test_mask_check_passes_when_exactly_the_agent_turns_are_trained() -> None:
    ids, mask = _render(ROW, {0, 1, 2})
    chk = check(ROW, segments(_Tok(), ids, mask))
    assert chk["ok"], chk["problems"]
    assert chk["trained_turns"] == chk["assistant_turns"] == 3


def test_mask_check_fails_on_a_missing_turn_or_trained_context() -> None:
    ids, mask = _render(ROW, {0, 2})
    assert not check(ROW, segments(_Tok(), ids, mask))["ok"]
    ids, mask = _render(ROW, {0, 1, 2})
    assert not check(ROW, segments(_Tok(), ids, [1] * len(mask)))["ok"]
