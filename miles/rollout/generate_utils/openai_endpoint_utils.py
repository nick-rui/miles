"""
Utilities for the OpenAI endpoint
"""

import asyncio
import logging
from argparse import Namespace

from miles.rollout.session.reply_utils import SamplesReply, decode_samples_reply
from miles.utils.http_utils import post, post_bytes_no_retry
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SESSION_REQUEST_TIMEOUT = 120


class OpenAIEndpointTracer:
    def __init__(self, router_url: str, session_id: str, session_server_instance_id: str | None = None):
        self.router_url = router_url
        self.session_id = session_id
        self.base_url = f"{router_url}/sessions/{session_id}"
        self.session_server_instance_id = session_server_instance_id

    @staticmethod
    async def create(args: Namespace):
        session_ip = getattr(args, "session_server_ip", None)
        session_port = getattr(args, "session_server_port", None)
        if not session_ip or not session_port:
            raise RuntimeError(
                "session_server_ip/session_server_port are not set. "
                "Pass --use-session-server to start the session server."
            )
        session_url = f"http://{session_ip}:{session_port}"
        session_server_instance_id = getattr(args, "session_server_instance_id", None)
        response = await post(f"{session_url}/sessions", {}, action="post")
        session_id = response["session_id"]
        return OpenAIEndpointTracer(
            router_url=session_url,
            session_id=session_id,
            session_server_instance_id=session_server_instance_id,
        )

    async def collect_samples(
        self, input_sample: Sample, *, multi_samples: bool, max_seq_len: int | None
    ) -> SamplesReply:
        """Fetch the worker-assembled training samples for this session.

        Single direct POST, no retries: a 5xx means the owning worker died and
        the session's records died with it (the supervisor's check() backstops),
        and a 422 is a deterministic assembly failure whose assertion text is
        the body — both must raise loudly, immediately. A timeout raises too
        (assembly is seconds server-side; the old records path silently ABORTed
        the sample on timeout and lost data). The session DELETE is attempted
        on every path, success or failure, matching the old cleanup semantics;
        a DELETE failure is only a warning.
        """
        try:
            payload = await post_bytes_no_retry(
                f"{self.base_url}/samples",
                {"multi_samples": multi_samples, "max_seq_len": max_seq_len},
                timeout=_SESSION_REQUEST_TIMEOUT,
            )
        finally:
            try:
                await asyncio.wait_for(
                    post(self.base_url, {}, action="delete"),
                    timeout=_SESSION_REQUEST_TIMEOUT,
                )
            except Exception as e:
                logger.warning(f"Failed to delete session {self.session_id} after collecting samples: {e}")

        return decode_samples_reply(payload, input_sample)
