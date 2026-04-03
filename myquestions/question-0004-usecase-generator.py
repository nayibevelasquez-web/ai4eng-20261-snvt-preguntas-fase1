import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPClassifier
import random

def generar_caso_de_uso_predecir_epoca_dinosaurio():
    n_train = random.randint(80, 120)
    n_test = random.randint(20, 40)
    
    def generate_data(n):
        return pd.DataFrame({
            'densidad_calcio': np.random.exponential(scale=2.0, size=n),
            'porosidad': np.random.uniform(0.1, 0.9, size=n),
            'profundidad_fosilizacion': np.random.normal(50, 15, size=n),
            'epoca': np.random.choice(['Triasico', 'Jurasico', 'Cretacico'], size=n)
        })
        
    df_train = generate_data(n_train)
    df_test = generate_data(n_test)
    target_col = 'epoca'
    
    input_data = {'df_train': df_train.copy(), 'df_test': df_test.copy(), 'target_col': target_col}
    
    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values
    X_test = df_test.drop(columns=[target_col]).values
    qt = QuantileTransformer(n_quantiles=10, random_state=42)
    X_train_t = qt.fit_transform(X_train)
    X_test_t = qt.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42)
    mlp.fit(X_train_t, y_train)
    probs = mlp.predict_proba(X_test_t)
    
    output_data = probs
    return input_data, output_data

if __name__ == "__main__":
    in_data, out_data = generar_caso_de_uso_predecir_epoca_dinosaurio()
    print("Entrada (input_data keys):", in_data.keys())
    print("Salida esperada (output_data):\n", out_data[:2], "\n... (mostrando primeros 2)")
