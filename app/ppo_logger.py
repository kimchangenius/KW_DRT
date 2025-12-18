import csv
import os


def log_ppo_metrics(log_dir, episode, metrics):
    """
    PPO 학습 메트릭을 CSV로 누적 기록.
    metrics 예시:
    {
        "train_calls": int,
        "buffer_size": int,
        "train_updates": int,
        "avg_kl": float,
        "actor_loss": float,
        "critic_loss": float,
        "entropy": float,
        "clip_fraction": float,
        "explained_variance": float,
    }
    """
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, "ppo_train_log.csv")

    header = [
        "Episode",
        "TrainCalls",
        "BufferSize",
        "TrajectoryBufferSize",
        "TrainUpdates",
        "AvgKL",
        "ActorLoss",
        "CriticLoss",
        "Entropy",
        "ClipFraction",
        "ExplainedVariance",
    ]

    file_exists = os.path.exists(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            episode,
            metrics.get("train_calls", 0),
            metrics.get("buffer_size", 0),
            metrics.get("trajectory_buffer_size", 0),
            metrics.get("train_updates", 0),
            f"{metrics.get('avg_kl', 0.0):.6f}",
            f"{metrics.get('actor_loss', 0.0):.6f}",
            f"{metrics.get('critic_loss', 0.0):.6f}",
            f"{metrics.get('entropy', 0.0):.6f}",
            f"{metrics.get('clip_fraction', 0.0):.6f}",
            f"{metrics.get('explained_variance', 0.0):.6f}",
        ])

