"""Install feedback on the pinned vLLM V1 native Suffix worker."""

import vllm
from suffix_feedback import attach_feedback
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.worker.gpu_worker import Worker


class SuffixFeedbackWorker(Worker):
    def init_device(self):
        super().init_device()
        if vllm.__version__ != "0.32.0" or self.use_v2_model_runner:
            raise RuntimeError("Suffix feedback requires vLLM 0.32.0 and its V1 runner")
        runner = self.model_runner
        if not isinstance(runner.drafter, SuffixDecodingProposer):
            raise TypeError("Suffix feedback requires the native Suffix proposer")
        attach_feedback(runner, runner.drafter.num_speculative_tokens)
