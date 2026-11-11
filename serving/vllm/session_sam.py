"""SAM-Decoding retrieval with request-local feedback from AgSpec Eq. (3)."""

import json
import math
from dataclasses import dataclass, field


@dataclass
class Node:
    length: int
    link: int
    first_end: int
    next: dict = field(default_factory=dict)


class SamTape:
    """Suffix automaton with the earliest continuation for each match."""

    def __init__(self, tokens=()):
        self.tokens = []
        self.nodes = [Node(0, -1, -1)]
        self.last = 0
        self.extend(tokens)

    def extend(self, tokens):
        nodes = self.nodes
        for token in tokens:
            position = len(self.tokens)
            self.tokens.append(token)
            previous = self.last
            current = len(nodes)
            nodes.append(Node(nodes[previous].length + 1, 0, position))
            self.last = current
            while previous >= 0 and token not in nodes[previous].next:
                nodes[previous].next[token] = current
                previous = nodes[previous].link
            if previous < 0:
                continue
            target = nodes[previous].next[token]
            if nodes[previous].length + 1 == nodes[target].length:
                nodes[current].link = target
                continue
            clone = len(nodes)
            other = nodes[target]
            nodes.append(
                Node(nodes[previous].length + 1, other.link, other.first_end, other.next.copy())
            )
            while previous >= 0 and nodes[previous].next.get(token) == target:
                nodes[previous].next[token] = clone
                previous = nodes[previous].link
            nodes[current].link = other.link = clone

    def continuation(self, suffix, cap):
        nodes = self.nodes
        state = length = 0
        for token in suffix:
            while state and token not in nodes[state].next:
                state = nodes[state].link
                length = min(length, nodes[state].length)
            if token in nodes[state].next:
                state = nodes[state].next[token]
                length += 1
            else:
                length = 0
        while length >= 2:
            start = nodes[state].first_end + 1
            if start < len(self.tokens):
                return length, self.tokens[start : start + cap]
            state = nodes[state].link
            length = min(length, nodes[state].length)
        return 0, []


@dataclass
class RequestDraft:
    histories: list[SamTape]
    current: SamTape = field(default_factory=SamTape)
    scale: float = 1.0


class SamProposer:
    def __init__(self, config):
        from vllm.tokenizers import cached_tokenizer_from_config

        self.tokenizer = cached_tokenizer_from_config(model_config=config.model_config)
        self.cap = config.speculative_config.num_speculative_tokens
        self.states: dict[str, RequestDraft] = {}

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *, slot_mappings):
        runner = self.runner
        for key in self.states.keys() - runner.requests.keys():
            del self.states[key]
        proposals = []
        for i, key in enumerate(runner.input_batch.req_ids):
            request = runner.requests[key]
            params = request.sampling_params
            meta = (params.extra_args or {}).get("agentserve_v5_sam")
            sampled = sampled_token_ids[i]
            if meta is None or not sampled:
                proposals.append([])
                continue
            size = int(num_tokens_no_spec[i])
            room = min(
                self.cap,
                params.max_tokens - (size - request.num_prompt_tokens),
                runner.max_model_len - size,
            )
            if room <= 0:
                proposals.append([])
                continue
            if key not in self.states:
                self.states[key] = RequestDraft(
                    [
                        SamTape(self.tokenizer.encode(answer, add_special_tokens=False))
                        for answer in json.loads(meta)
                    ]
                )
            state = self.states[key]
            state.current.extend(token_ids_cpu[i, len(state.current.tokens) : size].tolist())
            actual = self.verified[i] if self.verified else 0
            if actual:
                accepted = min(actual, len(sampled) - 1)
                if accepted != self.cap or actual != self.cap:
                    state.scale = max(
                        0.125, min(2.0, state.scale + 0.1 * (0.9 - (accepted < actual)))
                    )
            suffix = state.current.tokens[-32:]
            length, draft = state.current.continuation(suffix, room)
            for history in state.histories:
                matched, candidate = history.continuation(suffix, room)
                if matched > length:
                    length, draft = matched, candidate
            proposals.append(draft[: max(1, math.floor(state.scale * length + 0.5))])
        return proposals
