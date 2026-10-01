# INFORME TÉCNICO Y ARQUITECTURA DEL PROYECTO
## Laboratorio 1: Minería de Datos (CRISP-DM) con Regresión Lineal Múltiple

---

## 1. 📁 ARQUITECTURA Y ESTRUCTURA DE CARPETAS DEL PROYECTO

El proyecto ha sido estructurado en módulos independientes organizados en carpetas específicas, siguiendo cada una de las fases fundamentales de la metodología **CRISP-DM**:

```
c:\Users\Admin\Desktop\mineria de datos\analisis dollar\
│
├── 📁 data/                           # Datos brutos y datos limpios
│   ├── dolar_data.csv
│   ├── glucosa_data.csv
│   ├── energia_data.csv
│   └── 📁 cleaned/                    # Datasets procesados y validados
│       ├── dolar_cleaned.csv
│       ├── glucosa_cleaned.csv
│       └── energia_cleaned.csv
│
├── 📁 1_limpieza_datos/               # FASE 1: Limpieza y Preparación de Datos
│   └── limpiar_datos.py
│
├── 📁 2_entrenamiento_modelos/        # FASE 2: Entrenamiento de Modelos
│   └── entrenar_modelos.py
│
├── 📁 3_analisis_evaluacion/          # FASE 3: Análisis de Métricas y Gráficos
│   └── analizar_resultados.py
│
├── 📁 4_interfaz_web/                 # FASE 4: Despliegue e Interfaz Web
│   └── app.py
│
├── 📁 modelos_exportados/             # Modelos serializados (.joblib)
│   ├── modelo_dolar.joblib
│   ├── modelo_glucosa.joblib
│   └── modelo_energia.joblib
│
├── 📁 resultados_graficos/            # Galería de imágenes y gráficos de resultados
│   ├── grafica_dolar.png
│   ├── grafica_glucosa.png
│   ├── grafica_energia.png
│   ├── matriz_correlacion.png
│   └── resultados_metricas.json
│
├── 📄 index.html                      # Dashboard HTML navegable con la galería de resultados
├── 📄 main.py                         # Orquestador del pipeline completo
└── 📄 INFORME_LABORATORIO_1.md        # Informe escrito técnico
```

---

## 2. 🔄 FASES DEL PIPELINE Y MÓDULOS

### 2.1 Módulo 1: Limpieza de Datos (`1_limpieza_datos/limpiar_datos.py`)
- **Objetivo:** Cargar los archivos CSV originales, auditar la presencia de valores nulos, eliminar duplicados e inspeccionar tipos de variables.
- **Entregable:** Genera los archivos saneados en `data/cleaned/`.

### 2.2 Módulo 2: Entrenamiento de Modelos (`2_entrenamiento_modelos/entrenar_modelos.py`)
- **Objetivo:** Ajustar modelos de Regresión Lineal Múltiple con `Scikit-Learn` sobre los conjuntos saneados.
- **Entregable:** Exporta los modelos ajustados en binarios `.joblib` en `modelos_exportados/`.

### 2.3 Módulo 3: Análisis y Evaluación (`3_analisis_evaluacion/analizar_resultados.py`)
- **Objetivo:** Evaluar las métricas de precisión ($MSE$, $RMSE$, $R^2$), calcular coeficientes estandarizados ($\beta^*$) para medir la importancia relativa de cada variable independiente y generar las representaciones gráficas.
- **Entregable:** Exporta los gráficos en alta resolución en `resultados_graficos/`.

### 2.4 Módulo 4: Interfaz Web (`4_interfaz_web/app.py`)
- **Objetivo:** Desplegar una aplicación web dinámica basada en Flask que carga los modelos `.joblib` y ofrece un formulario interactivo para predecir por teclado los tres escenarios.

### 2.5 Galería e Index Visual (`index.html`)
- **Objetivo:** Dashboard web con diseño moderno y responsivo que consolida las métricas globales y exhibe las imágenes de las gráficas de dispersión, líneas de tendencia y matrices de correlación de Pearson.

---

## 3. 📊 RESUMEN DE RESULTADOS Y MÉTRICAS OBTENIDAS

| Ejercicio | Variable Dependiente ($Y$) | Variables Independientes ($X$) | Intercepto ($\beta_0$) | Coeficiente Estandarizado Mayor ($\beta^*$) | $MSE$ | $RMSE$ | $R^2$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Precio Dólar** | `Precio_Dolar` | `Dia`, `Inflacion`, `Tasa_interes` | **3985.78** | **Dia** ($+721.55$) | 2,376.97 | **$48.75 COP** | **0.9963** (99.63%) |
| **2. Glucosa Sangre** | `Nivel_Glucosa` | `Edad`, `IMC`, `Actividad_Fisica` | **65.86** | **Edad** ($+21.64$) | 233.69 | **15.29 mg/dL** | **0.6814** (68.14%) |
| **3. Consumo Energía** | `Consumo_Energia`| `Temperatura`, `Hora`, `Dia_Semana`| **101.29** | **Temperatura** ($+49.87$) | 429.52 | **20.72 kWh** | **0.8968** (89.68%) |

---

## 4. 🚀 GUÍA DE EJECUCIÓN

1. **Ejecutar todo el pipeline en orden automático:**
   ```bash
   python main.py
   ```

2. **Visualizar el Index con la Galería de Imágenes:**
   - Abra el archivo [index.html](file:///c:/Users/Admin/Desktop/mineria%20de%20datos/analisis%20dollar/index.html) directamente en su navegador web.

3. **Iniciar la Interfaz Web de Predicción:**
   ```bash
   python 4_interfaz_web/app.py
   ```
   Acceda desde su navegador a: `http://127.0.0.1:5000`
