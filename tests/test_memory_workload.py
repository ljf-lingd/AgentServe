from agentserve.workloads.memory import build


def test_frozen_split_sizes_no_gold_in_runtime_and_determinism():
    inputs, gold = build("acceptance")
    assert len(inputs) == 36 and len(gold) == 864
    assert [len(c["turns"]) for c in inputs] == [12] * 24 + [48] * 12
    assert inputs == build("acceptance")[0]
    for conversation in inputs:
        for turn in conversation["turns"]:
            assert set(turn) == {"id", "question"}
            assert set(gold[turn["id"]]) == {"sql", "ordered", "tags", "clarify"}
    assert len(build("development")[0]) == 12
