**Table 9** — Final candidate configurations: GRU baseline vs. best EA-discovered pipeline.

| Parameter | GRU Baseline Official | Best Evolutionary GRU |
|---|---|---|
| n_layers | 2 | 2 |
| units1 | 96 | 64 |
| units2 | 64 | 64 |
| units3 | 96 | 32 |
| dropout | 0.0 | 0.0 |
| l2 | 0.000001 | 0.000010 |
| dense_units | 256 | 256 |
| dense_activation | relu | relu |
| learning_rate | 0.0002 | 0.0003 |
| clipnorm | 2.0 | 5.0 |
| optimizer_name | adamw | adamw |
| weight_decay | 0.000001 | 0.000000 |
| loss_name | huber1 | mae |
| gaussian_noise_std | 0.0 | 0.0 |
| batch_size | 128 | 256 |
| scaler_name | standard | robust |
| lookback | 120 | 144 |
