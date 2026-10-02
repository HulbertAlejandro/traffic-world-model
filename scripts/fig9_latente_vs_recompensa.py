"""Figura 8 del artículo (v6): MSE latente y de recompensa por horizonte. Lee results/*_evaluation.json."""

import json, matplotlib.pyplot as plt

def load(p):
    d = json.load(open(p))
    hs = sorted(int(k) for k in d)
    return hs, [d[str(h)]["model_latent_mse"] for h in hs], [d[str(h)]["model_reward_mse"] for h in hs]

h, lL, rL = load("results/world_model_evaluation.json")
_, lT, rT = load("results/world_model_transformer_evaluation.json")
_, lM, rM = load("results/world_model_tsmixer_evaluation.json")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(h, lL, "o-", label="LSTM")
ax[0].plot(h, lT, "s-", label="Transformer")
ax[0].plot(h, lM, "^-", label="TSMixer")
ax[0].set(xlabel="Horizonte (pasos)", ylabel="MSE latente", title="(a) Estado latente")
ax[0].legend()

ax[1].plot(h, rL, "o-", label="LSTM")
ax[1].plot(h, rT, "s-", label="Transformer")
ax[1].plot(h, rM, "^-", label="TSMixer")
ax[1].set(xlabel="Horizonte (pasos)", ylabel="MSE de recompensa", title="(b) Recompensa")
ax[1].legend()

plt.tight_layout()
plt.savefig("results/fig9_latente_vs_recompensa.png", dpi=150)