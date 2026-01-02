import csv
import os


def log_dqn_metrics(log_dir, episode, metrics):
    """
    DQN/DDQN 학습 메트릭을 CSV로 누적 기록.
    metrics 예시:
    {
        "avg_q": float,
        "episode_length": int,
        "discounted_return": float,
    }
    """
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, "dqn_train_log.csv")

    header = [
        "Episode",
        "AvgQ",
        "EpisodeLength",
        "DiscountedReturn",
    ]

    file_exists = os.path.exists(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            episode,
            f"{metrics.get('avg_q', 0.0):.6f}",
            metrics.get("episode_length", 0),
            f"{metrics.get('discounted_return', 0.0):.6f}",
        ])

