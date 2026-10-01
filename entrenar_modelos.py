import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

# Configuration for plots
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

output_dir = "archivos_generados"
os.makedirs(output_dir, exist_ok=True)

results = {}

# ==========================================
# EJERCICIO 1: Precio del Dólar
# ==========================================
print("--- EJERCICIO 1: PREDICCIÓN PRECIO DEL DÓLAR ---")
df_dolar = pd.read_csv("dolar_data.csv")

X_dolar = df_dolar[['Dia', 'Inflacion', 'Tasa_interes']]
y_dolar = df_dolar['Precio_Dolar']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_dolar, y_dolar, test_size=0.2, random_state=42)

model_dolar = LinearRegression()
model_dolar.fit(X_train_d, y_train_d)

y_pred_dolar = model_dolar.predict(X_test_d)
mse_dolar = mean_squared_error(y_test_d, y_pred_dolar)
rmse_dolar = np.sqrt(mse_dolar)
r2_dolar = r2_score(y_test_d, y_pred_dolar)

model_dolar_full = LinearRegression()
model_dolar_full.fit(X_dolar, y_dolar)
joblib.dump(model_dolar_full, os.path.join(output_dir, "modelo_dolar.joblib"))

scaler_d = StandardScaler()
X_dolar_scaled = scaler_d.fit_transform(X_dolar)
model_dolar_std = LinearRegression().fit(X_dolar_scaled, y_dolar)

print(f"Intercepto: {model_dolar.intercept_:.4f}")
for col, coef in zip(X_dolar.columns, model_dolar.coef_):
    print(f"Coeficiente {col}: {coef:.4f}")
print(f"MSE: {mse_dolar:.4f}")
print(f"RMSE: {rmse_dolar:.4f}")
print(f"R²: {r2_dolar:.4f}\n")

# Plot Exercise 1
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Ejercicio 1: Variables Independientes vs Precio del Dólar', fontsize=14, fontweight='bold')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, col in enumerate(X_dolar.columns):
    x_val = df_dolar[col]
    y_val = df_dolar['Precio_Dolar']
    axes[i].scatter(x_val, y_val, alpha=0.5, color=colors[i], label='Datos')
    m, b = np.polyfit(x_val, y_val, 1)
    axes[i].plot(x_val, m*x_val + b, color='red', linewidth=2, label='Tendencia')
    axes[i].set_title(f'{col} vs Precio Dólar')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Precio Dólar ($)')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grafica_dolar.png"), dpi=300)
plt.close()

results['dolar'] = {
    'intercept': float(model_dolar.intercept_),
    'coefs': {k: float(v) for k, v in zip(X_dolar.columns, model_dolar.coef_)},
    'coefs_std': {k: float(v) for k, v in zip(X_dolar.columns, model_dolar_std.coef_)},
    'mse': float(mse_dolar),
    'rmse': float(rmse_dolar),
    'r2': float(r2_dolar),
    'n_samples': int(len(df_dolar))
}

# ==========================================
# EJERCICIO 2: Niveles de Glucosa
# ==========================================
print("--- EJERCICIO 2: PREDICCIÓN DE NIVELES DE GLUCOSA EN SANGRE ---")
df_glucosa = pd.read_csv("glucosa_data.csv")

X_glucosa = df_glucosa[['Edad', 'IMC', 'Actividad_Fisica']]
y_glucosa = df_glucosa['Nivel_Glucosa']

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_glucosa, y_glucosa, test_size=0.2, random_state=42)

model_glucosa = LinearRegression()
model_glucosa.fit(X_train_g, y_train_g)

y_pred_glucosa = model_glucosa.predict(X_test_g)
mse_glucosa = mean_squared_error(y_test_g, y_pred_glucosa)
rmse_glucosa = np.sqrt(mse_glucosa)
r2_glucosa = r2_score(y_test_g, y_pred_glucosa)

model_glucosa_full = LinearRegression()
model_glucosa_full.fit(X_glucosa, y_glucosa)
joblib.dump(model_glucosa_full, os.path.join(output_dir, "modelo_glucosa.joblib"))

scaler_g = StandardScaler()
X_glucosa_scaled = scaler_g.fit_transform(X_glucosa)
model_glucosa_std = LinearRegression().fit(X_glucosa_scaled, y_glucosa)

print(f"Intercepto: {model_glucosa.intercept_:.4f}")
for col, coef in zip(X_glucosa.columns, model_glucosa.coef_):
    print(f"Coeficiente {col}: {coef:.4f}")
print(f"MSE: {mse_glucosa:.4f}")
print(f"RMSE: {rmse_glucosa:.4f}")
print(f"R²: {r2_glucosa:.4f}\n")

# Plot Exercise 2
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Ejercicio 2: Variables Independientes vs Nivel de Glucosa', fontsize=14, fontweight='bold')
colors = ['#9467bd', '#8c564b', '#e377c2']

for i, col in enumerate(X_glucosa.columns):
    x_val = df_glucosa[col]
    y_val = df_glucosa['Nivel_Glucosa']
    axes[i].scatter(x_val, y_val, alpha=0.4, color=colors[i], label='Datos')
    m, b = np.polyfit(x_val, y_val, 1)
    axes[i].plot(x_val, m*x_val + b, color='darkblue', linewidth=2, label='Tendencia')
    axes[i].set_title(f'{col} vs Nivel Glucosa')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Nivel Glucosa (mg/dL)')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grafica_glucosa.png"), dpi=300)
plt.close()

results['glucosa'] = {
    'intercept': float(model_glucosa.intercept_),
    'coefs': {k: float(v) for k, v in zip(X_glucosa.columns, model_glucosa.coef_)},
    'coefs_std': {k: float(v) for k, v in zip(X_glucosa.columns, model_glucosa_std.coef_)},
    'mse': float(mse_glucosa),
    'rmse': float(rmse_glucosa),
    'r2': float(r2_glucosa),
    'n_samples': int(len(df_glucosa))
}

# ==========================================
# EJERCICIO 3: Consumo de Energía Eléctrica
# ==========================================
print("--- EJERCICIO 3: PREDICCIÓN DEL CONSUMO DE ENERGÍA ELÉCTRICA ---")
df_energia = pd.read_csv("energia_data.csv")

X_energia = df_energia[['Temperatura', 'Hora', 'Dia_Semana']]
y_energia = df_energia['Consumo_Energia']

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_energia, y_energia, test_size=0.2, random_state=42)

model_energia = LinearRegression()
model_energia.fit(X_train_e, y_train_e)

y_pred_energia = model_energia.predict(X_test_e)
mse_energia = mean_squared_error(y_test_e, y_pred_energia)
rmse_energia = np.sqrt(mse_energia)
r2_energia = r2_score(y_test_e, y_pred_energia)

model_energia_full = LinearRegression()
model_energia_full.fit(X_energia, y_energia)
joblib.dump(model_energia_full, os.path.join(output_dir, "modelo_energia.joblib"))

scaler_e = StandardScaler()
X_energia_scaled = scaler_e.fit_transform(X_energia)
model_energia_std = LinearRegression().fit(X_energia_scaled, y_energia)

print(f"Intercepto: {model_energia.intercept_:.4f}")
for col, coef in zip(X_energia.columns, model_energia.coef_):
    print(f"Coeficiente {col}: {coef:.4f}")
print(f"MSE: {mse_energia:.4f}")
print(f"RMSE: {rmse_energia:.4f}")
print(f"R²: {r2_energia:.4f}\n")

# Plot Exercise 3
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Ejercicio 3: Variables Independientes vs Consumo de Energía', fontsize=14, fontweight='bold')
colors = ['#7f7f7f', '#bcbd22', '#17becf']

for i, col in enumerate(X_energia.columns):
    x_val = df_energia[col]
    y_val = df_energia['Consumo_Energia']
    axes[i].scatter(x_val, y_val, alpha=0.2, color=colors[i], label='Datos')
    m, b = np.polyfit(x_val, y_val, 1)
    axes[i].plot(x_val, m*x_val + b, color='purple', linewidth=2, label='Tendencia')
    axes[i].set_title(f'{col} vs Consumo Energía')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Consumo Energía (kWh)')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grafica_energia.png"), dpi=300)
plt.close()

# Correlation heatmaps using matplotlib imshow
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Matriz de Correlación de Pearson por Dataset', fontsize=15, fontweight='bold')

datasets = [(df_dolar, 'Dólar', 'Blues'), (df_glucosa, 'Glucosa', 'Greens'), (df_energia, 'Energía', 'Oranges')]
for idx, (df, name, cmap) in enumerate(datasets):
    corr = df.corr()
    cax = axes[idx].matshow(corr, cmap=cmap)
    fig.colorbar(cax, ax=axes[idx])
    axes[idx].set_xticks(range(len(corr.columns)))
    axes[idx].set_yticks(range(len(corr.columns)))
    axes[idx].set_xticklabels(corr.columns, rotation=45)
    axes[idx].set_yticklabels(corr.columns)
    axes[idx].set_title(f'Dataset {name}', pad=20)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            axes[idx].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matriz_correlacion.png"), dpi=300)
plt.close()

results['energia'] = {
    'intercept': float(model_energia.intercept_),
    'coefs': {k: float(v) for k, v in zip(X_energia.columns, model_energia.coef_)},
    'coefs_std': {k: float(v) for k, v in zip(X_energia.columns, model_energia_std.coef_)},
    'mse': float(mse_energia),
    'rmse': float(rmse_energia),
    'r2': float(r2_energia),
    'n_samples': int(len(df_energia))
}

print("=== RESUMEN EJECUCIÓN COMPLETA ===")
print(json.dumps(results, indent=4))

with open(os.path.join(output_dir, "resultados_metricas.json"), "w") as f:
    json.dump(results, f, indent=4)

print("¡Proceso completado exitosamente! Modelos y gráficos exportados en la carpeta 'archivos_generados'.")
