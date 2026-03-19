from __future__ import annotations

import numpy as np

from lwmg.evaluation.interpretability import latent_pca


if __name__ == "__main__":
    latents = np.random.randn(100, 16)
    proj = latent_pca(latents)
    print("latent pca shape:", proj.shape)
