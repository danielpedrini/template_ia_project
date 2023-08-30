def ml_regressionevaluation(y_test, y_predict, X_test):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from math import sqrt
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    k = X_test.shape[1]
    n = len(X_test)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2:', r2)
    print('R2 Ajustado:', adj_r2)