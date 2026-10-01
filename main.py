import sys
import os
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mod_limpieza = import_module_from_path("limpiar_datos", os.path.join(BASE_DIR, "1_limpieza_datos", "limpiar_datos.py"))
mod_entrenamiento = import_module_from_path("entrenar_modelos", os.path.join(BASE_DIR, "2_entrenamiento_modelos", "entrenar_modelos.py"))
mod_analisis = import_module_from_path("analizar_resultados", os.path.join(BASE_DIR, "3_analisis_evaluacion", "analizar_resultados.py"))

def ejecutar_pipeline_completo():
    print("==========================================================================")
    print(" PIPELINE COMPLETO - LABORATORIO 1 MINERIA DE DATOS (CRISP-DM)")
    print("==========================================================================")
    
    # 1. Limpieza de datos
    mod_limpieza.cargar_y_limpiar_datos()
    
    # 2. Entrenamiento y exportacion de modelos
    mod_entrenamiento.entrenar_y_exportar_modelos()
    
    # 3. Analisis, metricas y graficos
    mod_analisis.analizar_y_evaluar_modelos()
    
    print("\n==========================================================================")
    print(" PIPELINE EJECUTADO CON EXITO!")
    print(" - Modelos exportados en: modelos_exportados/")
    print(" - Graficos e imagenes en: resultados_graficos/")
    print(" - Abra 'index.html' en su navegador para ver la galeria completa de resultados.")
    print("==========================================================================")

if __name__ == "__main__":
    ejecutar_pipeline_completo()
