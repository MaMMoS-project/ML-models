from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_data(X_train, X_test, scaler_type='standard'):
    """Scales input data using the specified scaler type."""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    if scaler_type not in scalers:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")

    scaler = scalers[scaler_type]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

