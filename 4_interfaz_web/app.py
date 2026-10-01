from flask import Flask, render_template_string, request, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Search models in export folder
MODEL_DIR = os.path.join("..", "modelos_exportados")
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = "modelos_exportados"
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = "archivos_generados"

models = {}
for name in ['dolar', 'glucosa', 'energia']:
    path = os.path.join(MODEL_DIR, f"modelo_{name}.joblib")
    if os.path.exists(path):
        models[name] = joblib.load(path)
    elif os.path.exists(f"modelos_exportados/modelo_{name}.joblib"):
        models[name] = joblib.load(f"modelos_exportados/modelo_{name}.joblib")
    elif os.path.exists(f"archivos_generados/modelo_{name}.joblib"):
        models[name] = joblib.load(f"archivos_generados/modelo_{name}.joblib")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laboratorio Minería de Datos - Predicciones</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
            min-height: 100vh;
            padding: 30px 0;
        }
        .card-custom {
            background: rgba(30, 41, 59, 0.85);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 25px;
        }
        .card-header-custom {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-top-left-radius: 16px !important;
            border-top-right-radius: 16px !important;
            padding: 15px 25px;
            font-weight: 600;
        }
        .btn-predict {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 12px 25px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
            color: white;
        }
        .result-badge {
            font-size: 1.8rem;
            font-weight: 700;
            color: #38bdf8;
            padding: 15px;
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(56, 189, 248, 0.3);
            text-align: center;
        }
        .form-control, .form-select {
            background-color: #0f172a;
            border: 1px solid #334155;
            color: #f8fafc;
        }
        .form-control:focus, .form-select:focus {
            background-color: #0f172a;
            color: #f8fafc;
            border-color: #38bdf8;
            box-shadow: 0 0 0 0.25rem rgba(56, 189, 248, 0.25);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="text-center mb-5">
            <h1 class="display-4 fw-bold text-primary">📊 Laboratorio de Minería de Datos</h1>
            <p class="lead text-light">Módulo de Despliegue e Interfaz Predictiva (CRISP-DM)</p>
            <a href="/index.html" class="btn btn-outline-info btn-sm">🔙 Ir al Dashboard Principal (Index con Resultados)</a>
        </div>

        <ul class="nav nav-pills nav-justified mb-4 card-custom p-2" id="labTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active text-white" id="dolar-tab" data-bs-toggle="pill" data-bs-target="#dolar" type="button" role="tab">💵 Ejercicio 1: Precio Dólar</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link text-white" id="glucosa-tab" data-bs-toggle="pill" data-bs-target="#glucosa" type="button" role="tab">🩸 Ejercicio 2: Glucosa Sangre</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link text-white" id="energia-tab" data-bs-toggle="pill" data-bs-target="#energia" type="button" role="tab">⚡ Ejercicio 3: Consumo Energía</button>
            </li>
        </ul>

        <div class="tab-content" id="labTabsContent">
            <!-- TAB DOLAR -->
            <div class="tab-pane fade show active" id="dolar" role="tabpanel">
                <div class="card card-custom">
                    <div class="card-header card-header-custom">💵 Predicción de Precio del Dólar (COP)</div>
                    <div class="card-body p-4">
                        <form method="POST" action="/predict/dolar">
                            <div class="row g-3">
                                <div class="col-md-4">
                                    <label class="form-label">Día (Número de día)</label>
                                    <input type="number" name="dia" class="form-control" value="50" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Inflación Diaria (ej. 0.02)</label>
                                    <input type="number" step="0.0001" name="inflacion" class="form-control" value="0.0200" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Tasa de Interés Diaria (%)</label>
                                    <input type="number" step="0.01" name="tasa_interes" class="form-control" value="5.00" required>
                                </div>
                            </div>
                            <div class="mt-4">
                                <button type="submit" class="btn btn-predict">🚀 Predecir Dólar</button>
                            </div>
                        </form>
                        {% if pred_dolar is not none %}
                        <div class="mt-4 result-badge">
                            Resultado Estimado: ${{ "%.2f"|format(pred_dolar) }} COP
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>

            <!-- TAB GLUCOSA -->
            <div class="tab-pane fade" id="glucosa" role="tabpanel">
                <div class="card card-custom">
                    <div class="card-header card-header-custom" style="background: linear-gradient(90deg, #ec4899 0%, #db2777 100%);">🩸 Predicción de Glucosa en Sangre</div>
                    <div class="card-body p-4">
                        <form method="POST" action="/predict/glucosa">
                            <div class="row g-3">
                                <div class="col-md-4">
                                    <label class="form-label">Edad (Años)</label>
                                    <input type="number" name="edad" class="form-control" value="45" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Índice Masa Corporal (IMC)</label>
                                    <input type="number" step="0.1" name="imc" class="form-control" value="25.0" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Actividad Física (Horas/semana)</label>
                                    <input type="number" name="actividad" class="form-control" value="4" required>
                                </div>
                            </div>
                            <div class="mt-4">
                                <button type="submit" class="btn btn-predict" style="background: linear-gradient(90deg, #ec4899 0%, #db2777 100%);">🚀 Predecir Glucosa</button>
                            </div>
                        </form>
                        {% if pred_glucosa is not none %}
                        <div class="mt-4 result-badge" style="color: #f472b6; border-color: rgba(244, 114, 182, 0.3);">
                            Resultado Estimado: {{ "%.2f"|format(pred_glucosa) }} mg/dL
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>

            <!-- TAB ENERGIA -->
            <div class="tab-pane fade" id="energia" role="tabpanel">
                <div class="card card-custom">
                    <div class="card-header card-header-custom" style="background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);">⚡ Predicción de Consumo de Energía</div>
                    <div class="card-body p-4">
                        <form method="POST" action="/predict/energia">
                            <div class="row g-3">
                                <div class="col-md-4">
                                    <label class="form-label">Temperatura (°C)</label>
                                    <input type="number" step="0.1" name="temperatura" class="form-control" value="25.0" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Hora del día (1 a 24)</label>
                                    <input type="number" name="hora" class="form-control" value="14" min="1" max="24" required>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">Día de la semana (1=Lunes ... 7=Domingo)</label>
                                    <select name="dia_semana" class="form-select" required>
                                        <option value="1">1 - Lunes</option>
                                        <option value="2">2 - Martes</option>
                                        <option value="3">3 - Miércoles</option>
                                        <option value="4">4 - Jueves</option>
                                        <option value="5">5 - Viernes</option>
                                        <option value="6">6 - Sábado</option>
                                        <option value="7">7 - Domingo</option>
                                    </select>
                                </div>
                            </div>
                            <div class="mt-4">
                                <button type="submit" class="btn btn-predict" style="background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);">🚀 Predecir Energía</button>
                            </div>
                        </form>
                        {% if pred_energia is not none %}
                        <div class="mt-4 result-badge" style="color: #fbbf24; border-color: rgba(251, 191, 36, 0.3);">
                            Resultado Estimado: {{ "%.2f"|format(pred_energia) }} kWh
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, pred_dolar=None, pred_glucosa=None, pred_energia=None)

@app.route('/predict/dolar', methods=['POST'])
def predict_dolar():
    dia = float(request.form['dia'])
    inflacion = float(request.form['inflacion'])
    tasa = float(request.form['tasa_interes'])
    pred = None
    if 'dolar' in models:
        pred = models['dolar'].predict(pd.DataFrame([[dia, inflacion, tasa]], columns=['Dia', 'Inflacion', 'Tasa_interes']))[0]
    return render_template_string(HTML_TEMPLATE, pred_dolar=pred, pred_glucosa=None, pred_energia=None)

@app.route('/predict/glucosa', methods=['POST'])
def predict_glucosa():
    edad = float(request.form['edad'])
    imc = float(request.form['imc'])
    act = float(request.form['actividad'])
    pred = None
    if 'glucosa' in models:
        pred = models['glucosa'].predict(pd.DataFrame([[edad, imc, act]], columns=['Edad', 'IMC', 'Actividad_Fisica']))[0]
    return render_template_string(HTML_TEMPLATE, pred_dolar=None, pred_glucosa=pred, pred_energia=None)

@app.route('/predict/energia', methods=['POST'])
def predict_energia():
    temp = float(request.form['temperatura'])
    hora = float(request.form['hora'])
    dia_sem = float(request.form['dia_semana'])
    pred = None
    if 'energia' in models:
        pred = models['energia'].predict(pd.DataFrame([[temp, hora, dia_sem]], columns=['Temperatura', 'Hora', 'Dia_Semana']))[0]
    return render_template_string(HTML_TEMPLATE, pred_dolar=None, pred_glucosa=None, pred_energia=pred)

if __name__ == '__main__':
    print("Iniciando Interfaz Web Flask en http://127.0.0.1:5000...")
    app.run(port=5000, debug=True)
