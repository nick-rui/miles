# doc-dev: docs/developer/session-server-sample-assembly.md
"""Wire codec for the samples op: worker-assembled Samples -> reply bytes -> driver overlay.

Lives in its own session module, not in stdlib-only `ipc.py` (imported by
the torch-free router) and not in `worker.py` (the driver needs the decode
side without the registry/backend stack). The worker assembles on blank
`Sample()` templates (see `records_utils.py`) and only the COMPUTED_FIELDS
cross the wire; the driver overlays them onto deepcopies of its local input
sample. Overlay equivalence with the legacy driver-side pipeline requires
the input sample to carry dataclass defaults on the fields that pipeline
evolved in place — see `_assert_overlay_template_defaults`.
"""

import dataclasses
from copy import deepcopy

import numpy as np

from miles.rollout.session.ipc import decode_envelope, encode_envelope
from miles.utils.types import Sample

# Every Sample field is either COMPUTED by assembly from the records (crosses
# the wire) or belongs to the driver's input-sample TEMPLATE (never crosses;
# the driver overlay keeps its local deepcopy's value). Adding a Sample field
# without classifying it here fails at import time, not silently at training.
COMPUTED_FIELDS = (
    "tokens",
    "response",
    "response_length",
    "loss_mask",
    "rollout_log_probs",
    "rollout_routed_experts",
    "rollout_indexer_topk",
    "status",
    "weight_versions",
    "prefix_cache_info",
)
TEMPLATE_FIELDS = (
    "group_index",
    "index",
    "prompt",
    "multimodal_inputs",
    "multimodal_train_inputs",
    "label",
    "reward",
    "remove_sample",
    "teacher_log_probs",
    "opd_reverse_kl",
    "metadata",
    "generate_function_path",
    "train_metadata",
    "session_id",
    "non_generation_time",
    "spec_info",
)

_SAMPLE_FIELDS = {f.name for f in dataclasses.fields(Sample)}
assert set(COMPUTED_FIELDS) | set(TEMPLATE_FIELDS) == _SAMPLE_FIELDS and not set(COMPUTED_FIELDS) & set(
    TEMPLATE_FIELDS
), (
    "Sample fields drifted: every field must be classified as COMPUTED (crosses the samples wire) "
    "or TEMPLATE (stays on the driver's input sample). "
    f"Unclassified: {sorted(_SAMPLE_FIELDS - set(COMPUTED_FIELDS) - set(TEMPLATE_FIELDS))}, "
    f"unknown: {sorted((set(COMPUTED_FIELDS) | set(TEMPLATE_FIELDS)) - _SAMPLE_FIELDS)}, "
    f"overlap: {sorted(set(COMPUTED_FIELDS) & set(TEMPLATE_FIELDS))}"
)

# Fields carried as raw binary segments (dtype + shape in the JSON meta); the
# scalar/list computed fields ride in the JSON meta directly. Token ids and
# logprobs are re-materialized as Python lists on decode, exactly like the
# legacy JSON path (int64/f64 round-trips are lossless for both).
_SEGMENT_DTYPES = {"tokens": np.int64, "rollout_log_probs": np.float64}
_SEGMENT_FIELDS = ("tokens", "rollout_log_probs", "rollout_routed_experts", "rollout_indexer_topk")

_OPD_STUDENT_TOP_LOGPROBS_KEY = "opd_student_top_logprobs"


@dataclasses.dataclass
class SamplesReply:
    """Decoded `POST /sessions/{id}/samples` reply."""

    samples: list[Sample]
    session_metadata: dict
    empty_reason: str | None


def encode_samples_reply(samples: list[Sample], session_metadata: dict, empty_reason: str | None = None) -> bytes:
    """Worker side: pack assembled samples into one envelope (JSON meta + binary body)."""
    sample_metas = []
    segments: list[bytes] = []
    offset = 0
    for sample in samples:
        segment_meta = {}
        for name in _SEGMENT_FIELDS:
            value = getattr(sample, name)
            if value is None:
                segment_meta[name] = None
                continue
            arr = np.asarray(value, dtype=_SEGMENT_DTYPES.get(name))
            data = arr.tobytes()
            segment_meta[name] = {
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "offset": offset,
                "nbytes": len(data),
            }
            segments.append(data)
            offset += len(data)
        sample_metas.append(
            {
                "response": sample.response,
                "response_length": sample.response_length,
                "loss_mask": sample.loss_mask,
                "status": sample.status.value,
                "weight_versions": sample.weight_versions,
                "prefix_cache_info": sample.prefix_cache_info.to_dict(),
                "segments": segment_meta,
            }
        )
    meta = {"samples": sample_metas, "session_metadata": session_metadata, "empty_reason": empty_reason}
    return encode_envelope(meta, b"".join(segments))


def decode_samples_reply(payload: bytes, input_sample: Sample) -> SamplesReply:
    """Driver side: overlay each wire sample's computed fields onto a deepcopy of `input_sample`."""
    meta, body = decode_envelope(payload)
    if meta["samples"]:
        _assert_overlay_template_defaults(input_sample)
    samples = []
    for sample_meta in meta["samples"]:
        sample = deepcopy(input_sample)
        segment_meta = sample_meta["segments"]
        tokens = _read_segment(body, segment_meta["tokens"])
        log_probs = _read_segment(body, segment_meta["rollout_log_probs"])
        sample.tokens = tokens.tolist() if tokens is not None else []
        sample.rollout_log_probs = log_probs.tolist() if log_probs is not None else None
        sample.rollout_routed_experts = _read_segment(body, segment_meta["rollout_routed_experts"])
        sample.rollout_indexer_topk = _read_segment(body, segment_meta["rollout_indexer_topk"])
        sample.response = sample_meta["response"]
        sample.response_length = sample_meta["response_length"]
        sample.loss_mask = sample_meta["loss_mask"]
        sample.status = Sample.Status(sample_meta["status"])
        sample.weight_versions = list(sample_meta["weight_versions"])
        sample.prefix_cache_info = Sample.PrefixCacheInfo.from_dict(sample_meta["prefix_cache_info"])
        samples.append(sample)
    return SamplesReply(samples=samples, session_metadata=meta["session_metadata"], empty_reason=meta["empty_reason"])


def _read_segment(body: bytes, segment_meta: dict | None) -> np.ndarray | None:
    if segment_meta is None:
        return None
    start = segment_meta["offset"]
    arr = np.frombuffer(body[start : start + segment_meta["nbytes"]], dtype=segment_meta["dtype"])
    return arr.reshape(segment_meta["shape"])


def _assert_overlay_template_defaults(input_sample: Sample) -> None:
    """Overlay equivalence precondition (fail-loud).

    The legacy driver-side pipeline EVOLVED some fields of the input sample in
    place (`weight_versions` append, `prefix_cache_info` accumulate, merge sums
    `spec_info` across turns, `strip_last_output_tokens` trims
    `teacher_log_probs`/`opd_reverse_kl`/`metadata["opd_student_top_logprobs"]`),
    while the overlay REPLACES the computed fields and carries the template
    verbatim. The two agree exactly when the input sample holds dataclass
    defaults on those fields — true for every sample fresh from the data loader
    (and `reset_for_retry` restores it on framework retries).
    """
    assert input_sample.weight_versions == [], (
        f"input sample must not carry weight_versions (got {input_sample.weight_versions}); "
        "the legacy pipeline appended to it, the samples-wire overlay replaces it"
    )
    assert (
        input_sample.prefix_cache_info.to_dict() == Sample.PrefixCacheInfo().to_dict()
    ), f"input sample must carry a default prefix_cache_info (got {input_sample.prefix_cache_info.to_dict()})"
    assert (
        input_sample.spec_info.to_dict() == Sample.SpecInfo().to_dict()
    ), f"input sample must carry a default spec_info (got {input_sample.spec_info.to_dict()})"
    assert input_sample.teacher_log_probs is None and input_sample.opd_reverse_kl is None, (
        "input sample must not carry teacher_log_probs/opd_reverse_kl; "
        "the legacy pipeline trimmed them per turn, the samples-wire overlay carries them verbatim"
    )
    assert _OPD_STUDENT_TOP_LOGPROBS_KEY not in (input_sample.metadata or {}), (
        f"input sample metadata must not carry {_OPD_STUDENT_TOP_LOGPROBS_KEY!r}; "
        "merge_samples gives it per-token semantics that only hold for per-turn values"
    )
