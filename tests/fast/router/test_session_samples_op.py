"""OP_SAMPLES worker-op tests and the old-vs-new parity gate.

Drives `SessionWorker.handle` in-process against a real tokenizer (the
`test_session_worker.py` precedent), with records injected via the registry —
the broken-chain and R3 fixtures cannot be produced through the chat path.

The parity gate: the OLD pipeline is the literal driver-side sequence from
`agentic_tool_call.generate` (compute → agent metadata → truncate → merge →
session metadata); the NEW pipeline is OP_SAMPLES → `decode_samples_reply`
overlay → the same driver-side metadata application the client will do after
the cutover. Every `Sample` dataclass field must match.
"""

import dataclasses
import json
import uuid
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from fastapi.testclient import TestClient
from tests.fast.rollout.session.test_records_utils import _make_record

from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.session.core import build_session_core
from miles.rollout.session.ipc import (
    OP_CREATE,
    OP_HEALTH,
    OP_SAMPLES,
    decode_envelope,
    encode_envelope,
    encode_request,
)
from miles.rollout.session.records_utils import compute_samples_from_openai_records, truncate_samples_by_total_tokens
from miles.rollout.session.reply_utils import decode_samples_reply
from miles.rollout.session.router import build_router_app
from miles.rollout.session.worker import SessionWorker
from miles.utils.types import Sample

NUM_LAYERS = 3
TOPK = 2

_ARGS = SimpleNamespace(
    miles_router_timeout=30,
    hf_checkpoint="Qwen/Qwen3-0.6B",
    chat_template_path=None,
    apply_chat_template_kwargs={"enable_thinking": False},
    tito_model="default",
    tito_allowed_append_roles=["tool"],
    session_server_instance_id=uuid.uuid4().hex,
    num_layers=NUM_LAYERS,
    moe_router_topk=TOPK,
)


class _UnusedBackend:
    """OP_SAMPLES never proxies; any backend call is a test bug."""

    async def do_proxy(self, *args, **kwargs):
        raise AssertionError("collect_samples must not touch the proxy backend")


@pytest.fixture(scope="module")
def worker():
    return SessionWorker(build_session_core(_UnusedBackend(), _ARGS))


# ── fixtures: a two-turn trajectory with R3 / cache stats / weight versions ──


def _r3_b64(num_tokens: int, seed: int) -> str:
    arr = np.arange(seed, seed + num_tokens * NUM_LAYERS * TOPK, dtype=np.int32)
    return pybase64.b64encode(arr.tobytes()).decode("ascii")


def _two_turn_records():
    # R3 buffer length per record = (len(prompt) + len(output) - 1) * layers * topk.
    return [
        _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.125, -0.25],
            cached_tokens=0,
            prompt_tokens=3,
            weight_version="w1",
            routed_experts=_r3_b64(4, seed=0),
        ),
        _make_record(
            prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
            output_token_ids=[30, 31],
            output_log_probs=[-0.5, -1.0],
            cached_tokens=5,
            prompt_tokens=7,
            weight_version="w2",
            routed_experts=_r3_b64(8, seed=100),
        ),
    ]


_ACCUMULATED = [1, 2, 3, 10, 11, 20, 21, 30, 31]


def _input_sample() -> Sample:
    return Sample(
        group_index=4,
        index=9,
        prompt=[{"role": "user", "content": "hi"}],
        label="lbl",
        reward=2.5,
        metadata={"task": "t1", "shared_key": "from-input"},
        session_id="routing-sid",
        train_metadata={"loss": "ppo"},
        generate_function_path="gen.fn",
    )


# Overlapping keys lock the application order: agent overrides the input's
# shared_key; session_metadata (applied last) overrides the agent's
# max_trim_tokens plant.
_AGENT_METADATA = {"shared_key": "from-agent", "agent_only": 1, "max_trim_tokens": "agent-plant"}


async def _make_session(worker, records, accumulated) -> str:
    sid = uuid.uuid4().hex
    await worker.handle(encode_request(OP_CREATE, session_id=sid))
    session = worker.core.registry.sessions[sid]
    for record in records:
        session.append_record(record)
    if accumulated is not None:
        session.trajectory_token_ids.append(list(accumulated))
    return sid


async def _collect_via_op(worker, sid, *, multi_samples=False, max_seq_len=None):
    body = json.dumps({"multi_samples": multi_samples, "max_seq_len": max_seq_len}).encode()
    reply = await worker.handle(encode_request(OP_SAMPLES, session_id=sid, body=body))
    meta, payload = decode_envelope(reply)
    return meta["status"], payload


def _new_pipeline(payload, input_sample, *, multi_samples):
    """What collect_samples() does after the cutover: overlay + driver-side metadata."""
    reply = decode_samples_reply(payload, input_sample)
    samples = reply.samples
    for s in samples:
        s.metadata.update(_AGENT_METADATA)
    if samples:
        if not multi_samples:
            (merged,) = samples
            merged.metadata.update(reply.session_metadata)
        else:
            samples[-1].metadata.update(reply.session_metadata)
    return samples, reply


def _old_pipeline(worker, records, input_sample, *, multi_samples, max_seq_len, session_metadata):
    """agentic_tool_call.generate lines 96-129, verbatim semantics."""
    tokenizer = worker.core.registry.tokenizer
    samples = compute_samples_from_openai_records(
        _ARGS,
        input_sample,
        records,
        tokenizer,
        accumulated_token_ids=session_metadata.get("accumulated_token_ids"),
        max_trim_tokens=session_metadata.get("max_trim_tokens", 0),
    )
    for s in samples:
        s.metadata.update(_AGENT_METADATA)
    if max_seq_len is not None:
        samples = truncate_samples_by_total_tokens(samples, max_seq_len, tokenizer)
    if not samples:
        return []
    if not multi_samples:
        merged = merge_samples(samples, tokenizer)
        merged.metadata.update(session_metadata)
        return [merged]
    samples[-1].metadata.update(session_metadata)
    return samples


def _assert_samples_equal(old: list[Sample], new: list[Sample]):
    assert len(old) == len(new)
    for a, b in zip(old, new, strict=True):
        for f in dataclasses.fields(Sample):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if f.name in ("rollout_routed_experts", "rollout_indexer_topk"):
                if va is None or vb is None:
                    assert va is None and vb is None, f.name
                else:
                    assert va.dtype == vb.dtype, f.name
                    assert np.array_equal(va, vb), f.name
            elif f.name in ("spec_info", "prefix_cache_info"):
                assert va.to_dict() == vb.to_dict(), f.name
            else:
                assert va == vb, f"{f.name}: {va!r} != {vb!r}"


# ── the parity gate: {merge, multi_samples} x {truncation, none} ──


@pytest.mark.parametrize("multi_samples", [False, True], ids=["merge", "multi"])
@pytest.mark.parametrize("max_seq_len", [None, 8], ids=["no-trunc", "trunc"])
async def test_parity_old_vs_new(worker, multi_samples, max_seq_len):
    records = _two_turn_records()
    sid = await _make_session(worker, records, _ACCUMULATED)

    status, payload = await _collect_via_op(worker, sid, multi_samples=multi_samples, max_seq_len=max_seq_len)
    assert status == 200
    new_samples, reply = _new_pipeline(payload, _input_sample(), multi_samples=multi_samples)
    assert reply.empty_reason is None

    old_samples = _old_pipeline(
        worker,
        records,
        _input_sample(),
        multi_samples=multi_samples,
        max_seq_len=max_seq_len,
        session_metadata=reply.session_metadata,
    )
    _assert_samples_equal(old_samples, new_samples)
    if max_seq_len is not None:
        assert new_samples[-1].status == Sample.Status.TRUNCATED
        assert len(new_samples[-1].tokens) <= max_seq_len


async def test_session_metadata_matches_get_session(worker):
    """The samples reply and the records GET must expose the same metadata dict
    (both are built by the extracted _session_metadata helper)."""
    sid = await _make_session(worker, _two_turn_records(), _ACCUMULATED)
    _, payload = await _collect_via_op(worker, sid)
    reply = decode_samples_reply(payload, Sample())

    from miles.rollout.session.ipc import OP_GET

    meta, body = decode_envelope(await worker.handle(encode_request(OP_GET, session_id=sid)))
    assert meta["status"] == 200
    assert reply.session_metadata == json.loads(body)["metadata"]
    assert reply.session_metadata["accumulated_token_ids"] == _ACCUMULATED


# ── empty_reason discriminator ──


async def test_no_records_reply(worker):
    sid = await _make_session(worker, [], None)
    status, payload = await _collect_via_op(worker, sid)
    assert status == 200
    reply = decode_samples_reply(payload, Sample())
    assert reply.samples == [] and reply.empty_reason == "no_records"


async def test_all_truncated_reply(worker):
    # max_seq_len=2 < the first turn's prompt+1: truncate_samples_by_total_tokens
    # drops every turn -> empty samples with the all_truncated reason; the old
    # pipeline returns [] on the same fixture (today's ABORTED path).
    records = _two_turn_records()
    sid = await _make_session(worker, records, _ACCUMULATED)
    status, payload = await _collect_via_op(worker, sid, max_seq_len=2)
    assert status == 200
    reply = decode_samples_reply(payload, Sample())
    assert reply.samples == [] and reply.empty_reason == "all_truncated"
    assert (
        _old_pipeline(
            worker,
            records,
            _input_sample(),
            multi_samples=False,
            max_seq_len=2,
            session_metadata=reply.session_metadata,
        )
        == []
    )


# ── the 422 lane ──


async def test_broken_chain_returns_422_and_worker_survives(worker):
    # The accumulated sequence carries one token the records never produced ->
    # the cursor consistency assert fires -> 422 with the assertion text, and
    # the worker keeps serving (the failure never becomes an IPC ERROR frame).
    sid = await _make_session(worker, _two_turn_records(), _ACCUMULATED + [99])
    status, payload = await _collect_via_op(worker, sid)
    assert status == 422
    assert "cursor" in payload.decode()

    meta, _ = decode_envelope(await worker.handle(encode_request(OP_HEALTH)))
    assert meta["status"] == 200


async def test_missing_session_returns_404(worker):
    status, payload = await _collect_via_op(worker, uuid.uuid4().hex)
    assert status == 404
    assert "not found" in json.loads(payload)["error"]


# ── route registration order (the catch-all proxy must not swallow /samples) ──


class _RecordingChannel:
    def __init__(self):
        self.payloads = []

    async def request(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        return encode_envelope({"status": 200, "headers": {"content-type": "application/octet-stream"}}, b"ok")


def test_samples_route_registered_before_catch_all_proxy():
    channel = _RecordingChannel()
    client = TestClient(build_router_app([channel]))
    response = client.post("/sessions/abc123/samples", content=b'{"multi_samples":false,"max_seq_len":null}')
    assert response.status_code == 200

    meta, body = decode_envelope(channel.payloads[-1])
    assert meta["op"] == OP_SAMPLES, "catch-all session_proxy swallowed the samples route"
    assert meta["session_id"] == "abc123"
    assert body == b'{"multi_samples":false,"max_seq_len":null}'
