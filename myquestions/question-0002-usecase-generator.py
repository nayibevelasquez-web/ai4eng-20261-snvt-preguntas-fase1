import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
import random

def generar_caso_de_uso_evaluar_errores_cuanticos():
    n_samples = random.randint(40, 80)
    X = np.random.randn(n_samples, 3)
    mask = np.random.choice([True, False], size=X.shape, p=[0.1, 0.9])
    X[mask] = np.nan
    X_df = pd.DataFrame(X, columns=['temp_mk', 'campo_magnetico', 'ruido_termico'])
    y_array = np.random.uniform(0.01, 0.1, n_samples)
    
    input_data = {'X_df': X_df.copy(), 'y_array': y_array.copy()}
    
    imputer = KNNImputer(n_neighbors=2)
    X_imputed = imputer.fit_transform(X_df)
    svr = SVR(kernel='rbf', C=1.5)
    svr.fit(X_imputed, y_array)
    preds = svr.predict(X_imputed)
    max_err = float(np.max(np.abs(y_array - preds)))
    
    output_data = max_err
    return input_data, output_data

if __name__ == "__main__":
    in_data, out_data = generar_caso_de_uso_evaluar_errores_cuanticos()
    print("Entrada (input_data):", in_data)
    print("Salida esperada (output_data):", out_data)
