import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def entrenar_y_exportar_modelos():
    print("==================================================")
    print(" FASE 2: ENTRENAMIENTO DE MODELOS Y SERIALIZACIÓN")
    print("==================================================")
    
    cleaned_dir = os.path.join("data", "cleaned")
    export_dir = "modelos_exportados"
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Modelo Dólar
    print("\n[1/3] Entrenando Modelo Dólar...")
    df_dolar = pd.read_csv(os.path.join(cleaned_dir, "dolar_cleaned.csv"))
    X_dolar = df_dolar[['Dia', 'Inflacion', 'Tasa_interes']]
    y_dolar = df_dolar['Precio_Dolar']
    
    model_dolar = LinearRegression()
    model_dolar.fit(X_dolar, y_dolar)
    joblib.dump(model_dolar, os.path.join(export_dir, "modelo_dolar.joblib"))
    print(f"  - Modelo Dólar guardado en: {export_dir}/modelo_dolar.joblib")
    
    # 2. Modelo Glucosa
    print("\n[2/3] Entrenando Modelo Glucosa...")
    df_glucosa = pd.read_csv(os.path.join(cleaned_dir, "glucosa_cleaned.csv"))
    X_glucosa = df_glucosa[['Edad', 'IMC', 'Actividad_Fisica']]
    y_glucosa = df_glucosa['Nivel_Glucosa']
    
    model_glucosa = LinearRegression()
    model_glucosa.fit(X_glucosa, y_glucosa)
    joblib.dump(model_glucosa, os.path.join(export_dir, "modelo_glucosa.joblib"))
    print(f"  - Modelo Glucosa guardado en: {export_dir}/modelo_glucosa.joblib")
    
    # 3. Modelo Energía
    print("\n[3/3] Entrenando Modelo Energía...")
    df_energia = pd.read_csv(os.path.join(cleaned_dir, "energia_cleaned.csv"))
    X_energia = df_energia[['Temperatura', 'Hora', 'Dia_Semana']]
    y_energia = df_energia['Consumo_Energia']
    
    model_energia = LinearRegression()
    model_energia.fit(X_energia, y_energia)
    joblib.dump(model_energia, os.path.join(export_dir, "modelo_energia.joblib"))
    print(f"  - Modelo Energía guardado en: {export_dir}/modelo_energia.joblib")
    
    print("\n¡Fase 2 de Entrenamiento completada exitosamente!")

if __name__ == "__main__":
    entrenar_y_exportar_modelos()
