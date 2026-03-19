from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE


def latent_tsne(latents: np.ndarray, n_components: int = 2) -> np.ndarray:
    return TSNE(n_components=n_components, init="random", learning_rate="auto").fit_transform(latents)
