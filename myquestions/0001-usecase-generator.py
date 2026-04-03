import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import random

def generar_caso_de_uso_detectar_anomalias_murcielagos():
    n = random.randint(60, 100)
    df = pd.DataFrame({
        'frecuencia_pico': np.random.uniform(20.0, 120.0, n),
        'duracion_pulso': np.random.uniform(1.0, 15.0, n)
    })
    contaminacion = round(random.uniform(0.05, 0.15), 2)
    
    input_data = {'df': df.copy(), 'contaminacion': contaminacion}
    
    df_clean = df.copy()
    df_clean['frecuencia_pico'] = df_clean['frecuencia_pico'].rolling(window=3).mean()
    df_clean = df_clean.dropna()
    X = df_clean[['frecuencia_pico', 'duracion_pulso']].values
    iso = IsolationForest(contamination=contaminacion, random_state=42)
    preds = iso.fit_predict(X)
    
    output_data = preds
    return input_data, output_data

if __name__ == "__main__":
    in_data, out_data = generar_caso_de_uso_detectar_anomalias_murcielagos()
    print("Entrada (input_data):", in_data)
    print("Salida esperada (output_data):", out_data)
