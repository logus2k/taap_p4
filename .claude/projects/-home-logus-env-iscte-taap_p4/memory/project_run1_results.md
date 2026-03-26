---
name: First notebook run results
description: Baseline and EA results from first complete notebook run (2026-03-26)
type: project
---

Results from first complete run of final_project_scope_V4.ipynb:

**Test Set Metrics:**

| Model | MAE (scaled) | RMSE (scaled) | MAE (°C) | RMSE (°C) |
|---|---|---|---|---|
| Persistence | 0.3637 | 0.4921 | 3.14 | 4.25 |
| GRU Baseline (official) | 0.1926 | 0.2555 | 1.67 | 2.21 |

**EA Search (validation fitness):**
- Gen 1: 1.734 → Gen 2: 1.735 → Gen 5: 1.707
- Best validation fitness: 1.704°C MAE

**Best EA configuration:**
- n_layers=1, units1=192, dense_units=256, dense_activation=leaky_relu
- optimizer=adamw, weight_decay=1e-4, lr=2e-4, loss=huber1
- batch_size=256, clipnorm=5.0, gaussian_noise=0.05, scaler=standard
- Total params: 176,536 (vs baseline 86,744)

**TimeGAN:**
- Autoencoder reconstruction MSE: 0.0014
- Adversarial training: 15 epochs, d_loss ~1.2, g_loss ~3.2

**Why:** Reference for comparing second run results after ensemble and tf.function fix.

**How to apply:** Compare these numbers with the next run to verify consistency and measure ensemble improvement.