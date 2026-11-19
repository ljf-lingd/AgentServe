"""Request-local rejection feedback applied to native Suffix draft prefixes."""

import math
from functools import wraps


class DraftFeedback:
    def __init__(self, hard_cap):
        self.hard_cap = hard_cap
        self.scales = {}

    def observe(self, request_id, verified, accepted):
        if verified and not verified == accepted == self.hard_cap:
            scale = self.scales.get(request_id, 1.0)
            self.scales[request_id] = max(0.125, min(2.0, scale + 0.1 * (0.9 - (accepted < verified))))

    def prefix(self, request_id, draft):
        budget = max(1, math.floor(self.scales.get(request_id, 1.0) * len(draft) + 0.5))
        return draft[:min(self.hard_cap, budget)]

    def retain(self, requests):
        for request_id in self.scales.keys() - requests.keys():
            del self.scales[request_id]


def attach_feedback(runner, hard_cap):
    """Bind to the V1 runner hook that exposes actual verifier inputs."""
    feedback = DraftFeedback(hard_cap)
    propose = runner.drafter.propose
    propose_ids = runner.propose_draft_token_ids

    @wraps(propose)
    def trim(num_speculative_tokens, input_batch, sampled_token_ids, slot_mappings=None):
        drafts = propose(num_speculative_tokens, input_batch, sampled_token_ids, slot_mappings)
        return [feedback.prefix(key, draft)
                for key, draft in zip(input_batch.req_ids, drafts, strict=True)]

    @wraps(propose_ids)
    def observe(
        scheduler_output, sampled_token_ids, sampling_metadata, hidden_states,
        sample_hidden_states, aux_hidden_states, spec_decode_metadata, common_attn_metadata,
        slot_mappings,
    ):
        feedback.retain(runner.requests)
        if spec_decode_metadata is not None:
            for key, verified, sampled in zip(
                runner.input_batch.req_ids, spec_decode_metadata.num_draft_tokens,
                sampled_token_ids, strict=True,
            ):
                if sampled:
                    feedback.observe(key, verified, min(verified, len(sampled) - 1))
        return propose_ids(
            scheduler_output, sampled_token_ids, sampling_metadata, hidden_states,
            sample_hidden_states, aux_hidden_states, spec_decode_metadata, common_attn_metadata,
            slot_mappings,
        )

    runner.drafter.propose = trim
    runner.propose_draft_token_ids = observe
