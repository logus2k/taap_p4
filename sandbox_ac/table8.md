**Table 8** — Top candidate configurations discovered across all EA generations.

| Parameter | Candidate 1 | Candidate 2 | Candidate 3 | Candidate 4 | Candidate 5 |
|---|---|---|---|---|---|
| n_layers | 2 | 2 | 2 | 2 | 2 |
| units1 / units2 / units3 | 64 / 64 / 32 | 96 / 96 / 64 | 128 / 128 / 32 | 64 / 64 / 32 | 128 / 64 / 64 |
| dropout | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| L2 | 0.000010 | 0.000001 | 0.000100 | 0.000001 | 0.000001 |
| dense_units | 256 | 256 | 256 | 256 | 256 |
| dense_activation | relu | relu | relu | relu | relu |
| learning_rate | 0.0003 | 0.0003 | 0.0003 | 0.0003 | 0.0002 |
| batch_size | 256 | 256 | 256 | 256 | 128 |
| clipnorm | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| optimizer_name | adamw | adam | adam | adam | adam |
| weight_decay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| loss_name | mae | mae | mae | mae | mae |
| gaussian_noise_std | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| scaler_name | robust | robust | robust | robust | robust |
| lookback | 144 | 144 | 144 | 144 | 144 |
| **fitness_mae_degC** | **1.635730** | **1.635795** | **1.636511** | **1.637370** | **1.638159** |
| **rmse_degC** | **2.237388** | **2.229955** | **2.242214** | **2.239683** | **2.237043** |
| **mae_scaled** | **0.132586** | **0.132592** | **0.132650** | **0.132719** | **0.132783** |
| **rmse_scaled** | **0.181355** | **0.180752** | **0.181746** | **0.181541** | **0.181327** |
