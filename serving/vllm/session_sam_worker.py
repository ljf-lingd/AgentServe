"""Bind the SAM proposer to vLLM 0.32's V1 verification feedback."""

from functools import wraps

import vllm
from vllm.v1.worker.gpu_worker import Worker


class SessionSamWorker(Worker):
    def init_device(self):
        super().init_device()
        if vllm.__version__ != "0.32.0" or self.use_v2_model_runner:
            raise RuntimeError("AgentServe SAM requires vLLM 0.32.0 and its V1 runner")
        runner = self.model_runner
        drafter = runner.drafter
        drafter.runner = runner
        propose = runner.propose_draft_token_ids

        @wraps(propose)
        def with_feedback(
            scheduler_output,
            sampled_token_ids,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            common_attn_metadata,
            slot_mappings,
        ):
            drafter.verified = (
                spec_decode_metadata.num_draft_tokens if spec_decode_metadata is not None else ()
            )
            return propose(
                scheduler_output,
                sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                common_attn_metadata,
                slot_mappings,
            )

        runner.propose_draft_token_ids = with_feedback
