from src.features.scaling import get_scaler
from src.features.windowing import make_windows
from src.models.gru import build_gru_model
from src.models.train_eval import (
    train_model,
    evaluate_scaled_forecasts,
    evaluate_original_scale_forecasts,
    inverse_scale_target,
)


def evaluate_individual(
    cfg,
    df_train,
    df_val,
    final_feature_cols,
    target_idx,
    lookback=120,
    horizon=24,
    epochs=20,
    verbose=0,
):
    scaler = get_scaler(cfg["scaler_name"])

    X_train_df = df_train[final_feature_cols].copy()
    X_val_df = df_val[final_feature_cols].copy()

    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)

    X_train, y_train = make_windows(X_train_scaled, target_idx, lookback, horizon)
    X_val, y_val = make_windows(X_val_scaled, target_idx, lookback, horizon)

    model = build_gru_model(
        L=lookback,
        n_features=X_train.shape[2],
        H=horizon,
        units1=cfg["units1"],
        units2=cfg["units2"],
        units3=cfg["units3"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        l2=cfg["l2"],
        dense_units=cfg["dense_units"],
        dense_activation=cfg["dense_activation"],
        learning_rate=cfg["learning_rate"],
        clipnorm=cfg["clipnorm"],
        optimizer_name=cfg["optimizer_name"],
        weight_decay=cfg["weight_decay"],
        loss_name=cfg["loss_name"],
        gaussian_noise_std=cfg["gaussian_noise_std"],
    )

    train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=cfg["batch_size"],
        epochs=epochs,
        verbose=verbose,
    )

    y_pred_val = model.predict(X_val, verbose=0)

    scaled_metrics = evaluate_scaled_forecasts(y_val, y_pred_val)

    target_mean = scaler.mean_[target_idx] if hasattr(scaler, "mean_") else None
    target_scale = scaler.scale_[target_idx] if hasattr(scaler, "scale_") else None

    if target_mean is not None and target_scale is not None:
        y_val_inv = inverse_scale_target(y_val, target_mean, target_scale)
        y_pred_val_inv = inverse_scale_target(y_pred_val, target_mean, target_scale)
    else:
        data_min = scaler.data_min_[target_idx]
        data_max = scaler.data_max_[target_idx]

        y_val_inv = y_val * (data_max - data_min) + data_min
        y_pred_val_inv = y_pred_val * (data_max - data_min) + data_min

    original_metrics = evaluate_original_scale_forecasts(y_val_inv, y_pred_val_inv)

    return {
        "cfg": cfg,
        "fitness": original_metrics["mae"],
        "metrics": {
            "mae_scaled": scaled_metrics["mae_scaled"],
            "rmse_scaled": scaled_metrics["rmse_scaled"],
            "mae_degC": original_metrics["mae"],
            "rmse_degC": original_metrics["rmse"],
        },
    }