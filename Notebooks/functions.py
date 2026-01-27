import pandas as pd
import os
from sklearn.metrics import roc_auc_score, f1_score

def evaluar_y_guardar_csv(modelo, X_test, y_test, nombre_prueba, nombre_archivo='metricas_modelos.csv'):
    folder = 'Data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, nombre_archivo)

    # 1. Realizar predicciones
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None

    # 2. Calcular métricas
    auc = round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else 0
    f1 = round(f1_score(y_test, y_pred), 4)
    
    # 3. Obtenemos los nombres de las columnas usadas
    variables = str(list(X_test.columns))
    
    # 4. Crear el nuevo registro (DataFrame de una fila)
    nuevo_registro = pd.DataFrame([{
        'Nombre_Prueba': nombre_prueba,
        'AUC-ROC': auc,
        'F1-Score': f1,
        'Variables_Usadas': variables
    }])
    
    # Si el archivo no existe, lo crea con cabecera. Si existe, añade sin cabecera.
    if not os.path.isfile(file_path):
        nuevo_registro.to_csv(file_path, index=False)
    else:
        nuevo_registro.to_csv(file_path, mode='a', header=False, index=False)
    
    # 5. Visualización de control
    print(f"\n📊 RESULTADOS: {nombre_prueba}")
    print(f"AUC: {auc:.4f} | F1: {f1:.4f}")
    print(f"✅ Métricas de '{nombre_prueba}' añadidas a: {file_path}") 