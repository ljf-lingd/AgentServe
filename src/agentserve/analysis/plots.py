"""Figures derive their conditions and sample counts from recorded results."""

import statistics

from agentserve.analysis.report import group_conditions, read


def workload_plots(summary, root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = group_conditions(summary["performance_runs"])
    if groups:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fields = [("completed_tasks_per_second", "correct_tasks_per_second"), ("mean_jct_seconds",)]
        labels = []
        for index, (key, runs) in enumerate(groups.items()):
            labels.append(key)
            for axis, metrics in zip(axes, fields, strict=True):
                for metric, color, offset in zip(metrics, ["#2563eb", "#16a34a"], [-0.1, 0.1]):
                    values = [r[metric] for r in runs if r[metric] is not None]
                    if not values:
                        continue
                    x = index + offset
                    axis.scatter(
                        [x] * len(values),
                        values,
                        color=color,
                        s=24,
                        label=metric if index == 0 else None,
                    )
                    axis.plot([x - 0.08, x + 0.08], [statistics.mean(values)] * 2, color="#111827")
        for axis, label in zip(axes, ["Task/s", "Mean task JCT (s)"], strict=True):
            axis.set_xticks(range(len(labels)), labels, rotation=20)
            axis.set_xlabel("Observed condition (dots are repeated runs)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        fig.suptitle("Real GPU Qwen: recorded workload conditions")
        fig.tight_layout()
        fig.savefig(output / "load-baseline.png", dpi=180)
        plt.close(fig)

    proof = read(root / "gpu-model-ready.json")
    uuid = proof["owned_model_processes"][0]["gpu_uuid"]
    import json

    samples = [json.loads(line) for line in (root / "gpu-samples.jsonl").read_text().splitlines()]
    samples = [s for s in samples if "gpus" in s]
    first = samples[0]["wall_time"]
    rows = [
        (s["wall_time"] - first, next(g for g in s["gpus"] if g["uuid"] == uuid)) for s in samples
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot([t for t, _ in rows], [int(g["used_mib"]) / 1024 for _, g in rows])
    axes[0].set_ylabel("GPU used GiB")
    axes[1].plot([t for t, _ in rows], [int(g["utilization_percent"]) for _, g in rows])
    axes[1].set_ylabel("GPU utilization %")
    axes[1].set_xlabel("Seconds from first sample")
    fig.suptitle("nvidia-smi samples; includes other existing processes on the same GPU")
    fig.tight_layout()
    fig.savefig(output / "gpu-observation.png", dpi=180)
    plt.close(fig)


def task_timeline(root, output, performance_runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    candidate = None
    for case in performance_runs:
        for path in sorted((root / case["case"] / "tasks").glob("*.json")):
            record = read(path)
            if record["status"] == "COMPLETED":
                candidate, source = record, path.relative_to(root)
                break
        if candidate is not None:
            break
    if candidate is None:
        return
    accepted = candidate["task"]["created_at"]
    starts, segments = {}, []
    for event in candidate["events"]:
        kind = event["type"]
        if kind in {"MODEL_QUEUED", "MODEL_STARTED", "SQL_STARTED"}:
            starts[kind] = event["wall_time"]
        if kind == "MODEL_STARTED":
            segments.append(
                (
                    "Model slot wait",
                    starts["MODEL_QUEUED"] - accepted,
                    event["wall_time"] - starts["MODEL_QUEUED"],
                )
            )
        elif kind == "MODEL_COMPLETED":
            segments.append(
                (
                    "Model request",
                    starts["MODEL_STARTED"] - accepted,
                    event["wall_time"] - starts["MODEL_STARTED"],
                )
            )
        elif kind == "SQL_COMPLETED":
            segments.append(
                (
                    "SQL stage",
                    starts["SQL_STARTED"] - accepted,
                    event["wall_time"] - starts["SQL_STARTED"],
                )
            )
    colors = {"Model slot wait": "#f59e0b", "Model request": "#2563eb", "SQL stage": "#16a34a"}
    fig, axis = plt.subplots(figsize=(10, 2.8))
    for label, start, duration in segments:
        axis.broken_barh([(start, duration)], (0, 1), facecolors=colors[label])
    axis.legend(
        handles=[Patch(color=color, label=label) for label, color in colors.items()],
        loc="upper center",
        ncol=3,
    )
    axis.set_ylim(-0.3, 2)
    axis.set_xlim(left=0)
    axis.set_yticks([])
    axis.set_xlabel("Seconds from durable task acceptance")
    axis.set_title(f"Observed task: {source}; stages include bookkeeping", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "task-timeline.png", dpi=180)
    plt.close(fig)
