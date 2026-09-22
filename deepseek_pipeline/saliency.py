import numpy as np


def gaussian_smooth(scores, sigma=2.0):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional sequence")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not len(scores) or sigma == 0:
        return scores.copy()
    radius = max(1, int(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(scores, kernel, mode="full")[radius:radius + len(scores)]
