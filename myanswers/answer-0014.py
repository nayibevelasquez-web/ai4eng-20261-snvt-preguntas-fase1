import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def preparar_datos(df: pd.DataFrame, target_col: str):
    """
    Transforma los datos crudos de los sensores imputando nulos con la mediana 
    y escalando las características al rango [0, 1].
    """
    # 1. Separar las características (X_raw) y la columna objetivo (y)
    X_raw = df.drop(columns=[target_col])
    
    # Convertimos explícitamente a un array de NumPy con tipo string ('O' o 'str')
    # para alinearnos con lo que espera recibir el validador en la comparación
    y = np.array(df[target_col].values, dtype=object)
    
    # 2. Imputación por Mediana para rellenar los NaN
    imputer = SimpleImputer(strategy='median')
    X_filled = imputer.fit_transform(X_raw)
    
    # 3. Escalado Min-Max al rango [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_filled)
    
    # 4. Retornar la tupla (X, y)
    return X, y
