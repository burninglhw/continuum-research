import torch

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


class ForcedTokens(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}
        self.device = device

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        def make_state(params, prompt_ids, output_ids):
            expected = (params.extra_args or {}).get("forced_token_ids")
            return (expected, output_ids) if expected is not None else None

        process_dict_updates(self.requests, batch_update, make_state)

    def apply(self, logits):
        if not self.requests:
            return logits
        rows = []
        tokens = []
        for row, (expected, generated) in self.requests.items():
            assert len(generated) < len(expected), (len(generated), len(expected))
            rows.append(row)
            tokens.append(expected[len(generated)])
        row_indices = torch.tensor(rows, device=logits.device, dtype=torch.long)
        token_indices = torch.tensor(tokens, device=logits.device, dtype=torch.long)
        logits[row_indices] = float("-inf")
        logits[row_indices, token_indices] = 0
        return logits
