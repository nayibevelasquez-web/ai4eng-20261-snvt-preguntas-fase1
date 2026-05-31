import pandas as pd

def recomendar_estrategia_aprendizaje(df: pd.DataFrame):
    """
    Agrupa las actividades por estudiante, determina la actividad más frecuente
    y devuelve un diccionario con las recomendaciones pedagógicas.
    """
    resultado = {}
    
    # 1. Agrupar los datos por estudiante_id
    for estudiante_id, grupo in df.groupby("estudiante_id"):
        
        # 2. Obtener la actividad más frecuente (la primera alfabéticamente en caso de empate)
        actividad_dominante = grupo["tipo_actividad"].mode()[0]
        
        # 3. Asignar la recomendación según la actividad
        if actividad_dominante == "lectura":
            recomendacion = "reforzar con ejercicios prácticos"
        elif actividad_dominante == "practica":
            recomendacion = "reforzar con teoría"
        else:
            recomendacion = "combinar con actividades escritas"
            
        # 4. Guardar en el diccionario resultado
        resultado[int(estudiante_id)] = recomendacion
        
    return resultado
