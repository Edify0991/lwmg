from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def latent_pca(latents: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(latents)
