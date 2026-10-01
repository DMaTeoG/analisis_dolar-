import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Laboratorio Minería de Datos - Predicciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .result-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

MODEL_DIR = "archivos_generados"

@st.cache_resource
def load_models():
    models = {}
    path_dolar = os.path.join(MODEL_DIR, "modelo_dolar.joblib")
    path_glucosa = os.path.join(MODEL_DIR, "modelo_glucosa.joblib")
    path_energia = os.path.join(MODEL_DIR, "modelo_energia.joblib")
    
    if os.path.exists(path_dolar):
        models['dolar'] = joblib.load(path_dolar)
    if os.path.exists(path_glucosa):
        models['glucosa'] = joblib.load(path_glucosa)
    if os.path.exists(path_energia):
        models['energia'] = joblib.load(path_energia)
    return models

models = load_models()

st.title("📊 Laboratorio 1: Minería de Datos (CRISP-DM)")
st.subheader("Sistema de Predicción con Modelos de Regresión Lineal Múltiple")
st.markdown("---")

# Sidebar navigation
st.sidebar.title("⚙️ Navegación")
opcion = st.sidebar.radio(
    "Seleccione el Escenario de Predicción:",
    ["1. Predicción Precio del Dólar 💵", 
     "2. Predicción Niveles de Glucosa 🩸", 
     "3. Predicción Consumo de Energía ⚡"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Metodología CRISP-DM**
- Comprensión del Negocio
- Comprensión de Datos
- Preparación de Datos
- Modelado (Regresión Lineal)
- Evaluación (MSE, RMSE, R²)
- Despliegue (Interfaz Web)
""")

# ----------------------------------------------------
# ESCENARIO 1: DÓLAR
# ----------------------------------------------------
if "1. Predicción Precio del Dólar" in opcion:
    st.header("💵 Predicción del Precio del Dólar (COP)")
    st.write("Ingrese los valores requeridos para estimar el precio del dólar según inflación, tasa de interés y día transcurrido.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Parámetros de Entrada")
        dia = st.number_input("Día transcurrido (Dia):", min_value=1, max_value=1000, value=50, step=1)
        inflacion = st.number_input("Tasa de Inflación Diaria (ej. 0.02 = 2%):", min_value=0.0, max_value=0.2, value=0.0200, format="%.6f", step=0.001)
        tasa_interes = st.number_input("Tasa de Interés Diaria (ej. 5.0 = 5%):", min_value=0.0, max_value=20.0, value=5.00, step=0.1)
        
        btn_pred = st.button("🚀 Calcular Predicción Dólar", use_container_width=True)

    with col2:
        st.subheader("📈 Resultado de la Predicción")
        if 'dolar' in models:
            input_data = pd.DataFrame([[dia, inflacion, tasa_interes]], columns=['Dia', 'Inflacion', 'Tasa_interes'])
            prediccion = models['dolar'].predict(input_data)[0]
            
            st.markdown(f"""
            <div class="result-box">
                Precio Estímado del Dólar:<br>
                <span style="font-size: 38px; color: #4CAF50;">${prediccion:,.2f} COP</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### ℹ️ Interpretación")
            st.write(f"- Para el **Día {dia}**, con **Inflación {inflacion*100:.4f}%** y **Tasa de Interés {tasa_interes:.2f}%**.")
        else:
            st.warning("El modelo de Dólar no ha sido entrenado aún. Ejecute `python entrenar_modelos.py` primero.")

    st.markdown("---")
    img_path = os.path.join(MODEL_DIR, "grafica_dolar.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Visualización de Relaciones - Dataset Dólar", use_column_width=True)

# ----------------------------------------------------
# ESCENARIO 2: GLUCOSA
# ----------------------------------------------------
elif "2. Predicción Niveles de Glucosa" in opcion:
    st.header("🩸 Predicción de Niveles de Glucosa en Sangre")
    st.write("Ingrese los datos del paciente para estimar su nivel de glucosa en sangre (mg/dL).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Datos del Paciente")
        edad = st.number_input("Edad (años):", min_value=1, max_value=120, value=45, step=1)
        imc = st.number_input("Índice de Masa Corporal (IMC):", min_value=10.0, max_value=60.0, value=25.0, step=0.5)
        actividad = st.slider("Horas semanales de actividad física:", min_value=0, max_value=20, value=4, step=1)
        
        btn_pred = st.button("🚀 Calcular Nivel de Glucosa", use_container_width=True)

    with col2:
        st.subheader("🩸 Resultado de la Predicción")
        if 'glucosa' in models:
            input_data = pd.DataFrame([[edad, imc, actividad]], columns=['Edad', 'IMC', 'Actividad_Fisica'])
            prediccion = models['glucosa'].predict(input_data)[0]
            
            # Category display
            estado = "Normal" if prediccion < 140 else "Pre-diabetes/Elevado" if prediccion < 180 else "Diabetes/Muy Elevado"
            color_estado = "#4CAF50" if prediccion < 140 else "#FF9800" if prediccion < 180 else "#F44336"

            st.markdown(f"""
            <div class="result-box">
                Nivel Estímado de Glucosa:<br>
                <span style="font-size: 38px; color: {color_estado};">{prediccion:.2f} mg/dL</span><br>
                <small>Categoría estimada: {estado}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("El modelo de Glucosa no ha sido entrenado aún. Ejecute `python entrenar_modelos.py` primero.")

    st.markdown("---")
    img_path = os.path.join(MODEL_DIR, "grafica_glucosa.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Visualización de Relaciones - Dataset Glucosa", use_column_width=True)

# ----------------------------------------------------
# ESCENARIO 3: ENERGÍA
# ----------------------------------------------------
elif "3. Predicción Consumo de Energía" in opcion:
    st.header("⚡ Predicción del Consumo de Energía Eléctrica")
    st.write("Ingrese las condiciones ambientales y de tiempo para estimar el consumo eléctrico en kWh.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Variables Ambientales y Temporales")
        temperatura = st.number_input("Temperatura Ambiental (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
        hora = st.slider("Hora del día (1 a 24):", min_value=1, max_value=24, value=14, step=1)
        dia_semana_num = st.selectbox("Día de la semana:", 
                                     options=[(1, "Lunes"), (2, "Martes"), (3, "Miércoles"), (4, "Jueves"), 
                                              (5, "Viernes"), (6, "Sábado"), (7, "Domingo")],
                                     format_func=lambda x: x[1])
        dia_semana = dia_semana_num[0]
        
        btn_pred = st.button("🚀 Calcular Consumo Energía", use_container_width=True)

    with col2:
        st.subheader("⚡ Resultado de la Predicción")
        if 'energia' in models:
            input_data = pd.DataFrame([[temperatura, hora, dia_semana]], columns=['Temperatura', 'Hora', 'Dia_Semana'])
            prediccion = models['energia'].predict(input_data)[0]
            
            st.markdown(f"""
            <div class="result-box">
                Consumo Estímado de Energía:<br>
                <span style="font-size: 38px; color: #FFC107;">{prediccion:.2f} kWh</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("El modelo de Energía no ha sido entrenado aún. Ejecute `python entrenar_modelos.py` primero.")

    st.markdown("---")
    img_path = os.path.join(MODEL_DIR, "grafica_energia.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Visualización de Relaciones - Dataset Energía", use_column_width=True)

st.markdown("---")
st.caption("Laboratorio de Minería de Datos - Modelos de Regresión Lineal Múltiple © 2026")
