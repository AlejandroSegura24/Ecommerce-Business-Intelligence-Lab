import os
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

def evaluar_y_guardar_csv(modelo, X_test, y_test, nombre_prueba, nombre_archivo='metricas_modelos.csv'):

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(root_path, 'Data')
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, nombre_archivo)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
    
    auc = round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else 0
    f1 = round(f1_score(y_test, y_pred), 4)
    variables = str(list(X_test.columns))

    nuevo_registro = pd.DataFrame([{
        'Nombre_Prueba': nombre_prueba,
        'AUC-ROC': auc,
        'F1-Score': f1,
        'Variables_Usadas': variables
    }])

    if not os.path.isfile(file_path):
        nuevo_registro.to_csv(file_path, index=False)
    else:
        nuevo_registro.to_csv(file_path, mode='a', header=False, index=False)
    
    print(f"✅ Métrica registrada en: {file_path}")