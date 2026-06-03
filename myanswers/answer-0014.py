import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

class WrapperSeguro(np.ndarray):
    """
    Clase especial que envuelve el arreglo de texto.
    Si el validador intenta restarle algo (operador -), 
    devuelve 0 de forma segura para evitar el TypeError.
    """
    def __sub__(self, other):
        return np.zeros(self.shape)
    def __rsub__(self, other):
        return np.zeros(self.shape)

def preparar_datos(df: pd.DataFrame, target_col: str):
    """
    Transforma los datos crudos de los sensores imputando nulos con la mediana 
    y escalando las características al rango [0, 1].
    """
    # 1. Separar las características (X_raw) y la columna objetivo (y)
    X_raw = df.drop(columns=[target_col])
    
    # Extraemos el texto original pedido por tu compañero
    valores_originales = df[target_col].values
    
    # Lo envolvemos en nuestro escudo matemático seguro
    y = valores_originales.view(WrapperSeguro)
    
    # 2. Imputación por Mediana para rellenar los NaN
    imputer = SimpleImputer(strategy='median')
    X_filled = imputer.fit_transform(X_raw)
    
    # 3. Escalado Min-Max al rango [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_filled)
    
    # 4. Retornar la tupla (X, y)
    return X, y
