import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_poisson_deviance
import random

def generar_caso_de_uso_calcular_devianza_aerogel():
    n_samples = random.randint(50, 100)
    df = pd.DataFrame({
        'presion': np.random.uniform(1.0, 10.0, n_samples),
        'concentracion': np.random.uniform(1.0, 5.0, n_samples),
        'tiempo_enfriamiento': np.random.uniform(10.0, 50.0, n_samples),
        'rendimiento_optico': np.random.normal(5.0, 3.0, n_samples) 
    })
    
    input_data = {'df_materiales': df.copy(), 'target_name': 'rendimiento_optico'}
    
    df_calc = df.copy()
    y = df_calc['rendimiento_optico'].values
    y = np.where(y <= 0, 0.1, y)
    X = df_calc.drop(columns=['rendimiento_optico']).values
    model = Ridge(alpha=2.0, random_state=123)
    model.fit(X, y)
    preds = model.predict(X)
    preds = np.clip(preds, 0.001, None)
    dev = float(mean_poisson_deviance(y, preds))
    output_data = int(dev * 1000)
    
    return input_data, output_data

if __name__ == "__main__":
    in_data, out_data = generar_caso_de_uso_calcular_devianza_aerogel()
    print("Entrada (input_data):", in_data)
    print("Salida esperada (output_data):", out_data)
