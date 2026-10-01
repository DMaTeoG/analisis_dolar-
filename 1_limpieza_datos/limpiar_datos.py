import pandas as pd
import numpy as np
import os

def cargar_y_limpiar_datos():
    print("==================================================")
    print(" FASE 1: CARGA Y LIMPIEZA DE DATOS (DATA PREP)")
    print("==================================================")
    
    input_dir = "data"
    output_dir = os.path.join("data", "cleaned")
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        "dolar": "dolar_data.csv",
        "glucosa": "glucosa_data.csv",
        "energia": "energia_data.csv"
    }
    
    cleaned_paths = {}
    
    for key, filename in datasets.items():
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            filepath = filename  # Fallback to current dir if needed
            
        print(f"\nProcesando dataset: {filename}...")
        df = pd.read_csv(filepath)
        
        # 1. Check missing values
        nulls = df.isnull().sum().sum()
        print(f"  - Valores nulos encontrados: {nulls}")
        if nulls > 0:
            df = df.dropna()
            
        # 2. Check duplicates
        dups = df.duplicated().sum()
        print(f"  - Duplicados encontrados: {dups}")
        if dups > 0:
            df = df.drop_duplicates()
            
        # 3. Data type verification
        print(f"  - Registros procesados: {len(df)} filas, {len(df.columns)} columnas.")
        
        # Save clean version
        save_path = os.path.join(output_dir, f"{key}_cleaned.csv")
        df.to_csv(save_path, index=False)
        cleaned_paths[key] = save_path
        print(f"  -> Guardado en: {save_path}")
        
    print("\n¡Fase 1 de Limpieza completada con éxito!")
    return cleaned_paths

if __name__ == "__main__":
    cargar_y_limpiar_datos()
