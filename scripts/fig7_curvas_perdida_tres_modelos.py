"""Figura 6 del artículo (v6): curvas de pérdida de LSTM, Transformer y TSMixer. Lee results/*_loss.png."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (f, t) in zip(ax, [
    ("results/world_model_loss.png", "LSTM"),
    ("results/world_model_transformer_loss.png", "Transformer"),
    ("results/world_model_tsmixer_loss.png", "TSMixer"),
]):
    img = mpimg.imread(f)
    a.imshow(img, aspect="auto")
    a.set_title(t, fontsize=11)
    a.axis("off")

plt.tight_layout(pad=0.4)
plt.savefig("results/fig7_curvas_perdida_tres_modelos.png", dpi=150)