import pandas as pd
from sklearn.linear_model import LogisticRegression

def entrenar_clasificador(df: pd.DataFrame, target_col: str):
    """
    Separa las características de la columna objetivo y entrena 
    un modelo de Regresión Logística.
    """
    # 1. Separar X (predictoras) e y (objetivo)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Crear el modelo de Regresión Logística
    model = LogisticRegression()
    
    # 3. Entrenar el modelo con los datos
    model.fit(X, y)
    
    # 4. Devolver el modelo entrenado
    return model
