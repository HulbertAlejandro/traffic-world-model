"""Figura 4 del artículo (v6): curvas de pérdida del Autoencoder y del LSTM. Lee results/*_loss.png."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (f, t) in zip(ax, [
    ("results/autoencoder_loss.png", "(a) Autoencoder"),
    ("results/world_model_loss.png", "(b) Modelo temporal (LSTM)"),
]):
    img = mpimg.imread(f)
    a.imshow(img, aspect="auto")
    a.set_title(t, fontsize=11)
    a.axis("off")

plt.tight_layout(pad=0.4)
plt.savefig("results/fig4_perdidas_ae_lstm.png", dpi=170)