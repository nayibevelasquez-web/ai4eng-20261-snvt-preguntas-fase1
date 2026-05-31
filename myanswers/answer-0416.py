import pandas as pd
from sklearn.linear_model import LinearRegression

def entrenar_modelo_energia(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Crea una característica cuadrática, entrena un modelo de Regresión Línea
    y devuelve los coeficientes, el intercepto y el valor R^2 en un diccionario.
    """
    # Para evitar modificar el DataFrame original del validador por referencia
    df_copy = df.copy()
    
    # 1. Crear la nueva columna con el cuadrado de la variable predictora
    df_copy['x_cuadrada'] = df_copy[x_col] ** 2
    
    # 2. Definir la matriz de características X y el vector objetivo y
    X = df_copy[[x_col, 'x_cuadrada']]
    y = df_copy[y_col]
    
    # 3. Crear y entrenar el modelo de Regresión Lineal
    model = LinearRegression()
    model.fit(X, y)
    
    # 4. Estructurar el diccionario de salida exactamente como el generador
    output_dict = {
        'coeficientes': model.coef_,
        'intercepto': model.intercept_,
        'r2': model.score(X, y)
    }
    
    return output_dict
