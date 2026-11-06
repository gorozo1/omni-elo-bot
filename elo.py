"""Elo rating helpers for a single pairwise evaluation."""


def update_ratings(rating_a: float, rating_b: float, result_a: float, k_factor: float = 32) -> tuple[float, float]:
    """Return updated ratings; result_a is 1 for A, 0 for B, and 0.5 for a tie."""
    if result_a not in (0, 0.5, 1):
        raise ValueError("result_a must be 0, 0.5, or 1")
    if k_factor <= 0:
        raise ValueError("k_factor must be positive")

    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    change_a = k_factor * (result_a - expected_a)
    return rating_a + change_a, rating_b - change_a