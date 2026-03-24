from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def get_scaler(name: str = "standard"):
    name = name.lower()

    if name == "standard":
        return StandardScaler()
    elif name == "robust":
        return RobustScaler()
    elif name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler: {name}")