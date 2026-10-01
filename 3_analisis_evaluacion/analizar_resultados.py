import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import json

def analizar_y_evaluar_modelos():
    print("==================================================")
    print(" FASE 3: EVALUACIÓN DE MÉTRICAS Y ANÁLISIS")
    print("==================================================")
    
    cleaned_dir = os.path.join("data", "cleaned")
    output_img_dir = "resultados_graficos"
    os.makedirs(output_img_dir, exist_ok=True)
    
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    results = {}
    
    # ----------------------------------------------------
    # 1. EJERCICIO 1: DÓLAR
    # ----------------------------------------------------
    print("\n[1/3] Evaluando Dataset Dólar...")
    df_dolar = pd.read_csv(os.path.join(cleaned_dir, "dolar_cleaned.csv"))
    X_dolar = df_dolar[['Dia', 'Inflacion', 'Tasa_interes']]
    y_dolar = df_dolar['Precio_Dolar']
    
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_dolar, y_dolar, test_size=0.2, random_state=42)
    m_dolar = LinearRegression().fit(X_tr_d, y_tr_d)
    
    y_pred_d = m_dolar.predict(X_te_d)
    mse_d = mean_squared_error(y_te_d, y_pred_d)
    rmse_d = np.sqrt(mse_d)
    r2_d = r2_score(y_te_d, y_pred_d)
    
    # Standardized coefficients
    X_d_std = StandardScaler().fit_transform(X_dolar)
    m_d_std = LinearRegression().fit(X_d_std, y_dolar)
    
    results['dolar'] = {
        'intercept': float(m_dolar.intercept_),
        'coefs': {col: float(c) for col, c in zip(X_dolar.columns, m_dolar.coef_)},
        'coefs_std': {col: float(c) for col, c in zip(X_dolar.columns, m_d_std.coef_)},
        'mse': float(mse_d),
        'rmse': float(rmse_d),
        'r2': float(r2_d)
    }
    
    # Plot Dólar
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Ejercicio 1: Variables vs Precio Dólar', fontsize=14, fontweight='bold')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, col in enumerate(X_dolar.columns):
        x_v, y_v = df_dolar[col], df_dolar['Precio_Dolar']
        axes[i].scatter(x_v, y_v, alpha=0.5, color=colors[i], label='Datos')
        m_val, b_val = np.polyfit(x_v, y_v, 1)
        axes[i].plot(x_v, m_val*x_v + b_val, color='red', linewidth=2, label='Tendencia')
        axes[i].set_title(f'{col} vs Precio Dólar')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Precio Dólar ($)')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "grafica_dolar.png"), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 2. EJERCICIO 2: GLUCOSA
    # ----------------------------------------------------
    print("\n[2/3] Evaluando Dataset Glucosa...")
    df_glucosa = pd.read_csv(os.path.join(cleaned_dir, "glucosa_cleaned.csv"))
    X_glucosa = df_glucosa[['Edad', 'IMC', 'Actividad_Fisica']]
    y_glucosa = df_glucosa['Nivel_Glucosa']
    
    X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_glucosa, y_glucosa, test_size=0.2, random_state=42)
    m_glucosa = LinearRegression().fit(X_tr_g, y_tr_g)
    
    y_pred_g = m_glucosa.predict(X_te_g)
    mse_g = mean_squared_error(y_te_g, y_pred_g)
    rmse_g = np.sqrt(mse_g)
    r2_g = r2_score(y_te_g, y_pred_g)
    
    X_g_std = StandardScaler().fit_transform(X_glucosa)
    m_g_std = LinearRegression().fit(X_g_std, y_glucosa)
    
    results['glucosa'] = {
        'intercept': float(m_glucosa.intercept_),
        'coefs': {col: float(c) for col, c in zip(X_glucosa.columns, m_glucosa.coef_)},
        'coefs_std': {col: float(c) for col, c in zip(X_glucosa.columns, m_g_std.coef_)},
        'mse': float(mse_g),
        'rmse': float(rmse_g),
        'r2': float(r2_g)
    }
    
    # Plot Glucosa
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Ejercicio 2: Variables vs Nivel Glucosa', fontsize=14, fontweight='bold')
    colors = ['#9467bd', '#8c564b', '#e377c2']
    for i, col in enumerate(X_glucosa.columns):
        x_v, y_v = df_glucosa[col], df_glucosa['Nivel_Glucosa']
        axes[i].scatter(x_v, y_v, alpha=0.4, color=colors[i], label='Datos')
        m_val, b_val = np.polyfit(x_v, y_v, 1)
        axes[i].plot(x_v, m_val*x_v + b_val, color='darkblue', linewidth=2, label='Tendencia')
        axes[i].set_title(f'{col} vs Nivel Glucosa')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Nivel Glucosa (mg/dL)')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "grafica_glucosa.png"), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # 3. EJERCICIO 3: ENERGÍA
    # ----------------------------------------------------
    print("\n[3/3] Evaluando Dataset Energía...")
    df_energia = pd.read_csv(os.path.join(cleaned_dir, "energia_cleaned.csv"))
    X_energia = df_energia[['Temperatura', 'Hora', 'Dia_Semana']]
    y_energia = df_energia['Consumo_Energia']
    
    X_tr_e, X_te_e, y_tr_e, y_te_e = train_test_split(X_energia, y_energia, test_size=0.2, random_state=42)
    m_energia = LinearRegression().fit(X_tr_e, y_tr_e)
    
    y_pred_e = m_energia.predict(X_te_e)
    mse_e = mean_squared_error(y_te_e, y_pred_e)
    rmse_e = np.sqrt(mse_e)
    r2_e = r2_score(y_te_e, y_pred_e)
    
    X_e_std = StandardScaler().fit_transform(X_energia)
    m_e_std = LinearRegression().fit(X_e_std, y_energia)
    
    results['energia'] = {
        'intercept': float(m_energia.intercept_),
        'coefs': {col: float(c) for col, c in zip(X_energia.columns, m_energia.coef_)},
        'coefs_std': {col: float(c) for col, c in zip(X_energia.columns, m_e_std.coef_)},
        'mse': float(mse_e),
        'rmse': float(rmse_e),
        'r2': float(r2_e)
    }
    
    # Plot Energía
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Ejercicio 3: Variables vs Consumo Energía', fontsize=14, fontweight='bold')
    colors = ['#7f7f7f', '#bcbd22', '#17becf']
    for i, col in enumerate(X_energia.columns):
        x_v, y_v = df_energia[col], df_energia['Consumo_Energia']
        axes[i].scatter(x_v, y_v, alpha=0.2, color=colors[i], label='Datos')
        m_val, b_val = np.polyfit(x_v, y_v, 1)
        axes[i].plot(x_v, m_val*x_v + b_val, color='purple', linewidth=2, label='Tendencia')
        axes[i].set_title(f'{col} vs Consumo Energía')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Consumo Energía (kWh)')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "grafica_energia.png"), dpi=300)
    plt.close()
    
    # Matrix of Correlation
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
    plt.savefig(os.path.join(output_img_dir, "matriz_correlacion.png"), dpi=300)
    plt.close()
    
    # Also sync graphics to root archivos_generados for legacy apps
    os.makedirs("archivos_generados", exist_ok=True)
    with open(os.path.join(output_img_dir, "resultados_metricas.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n- Métricas guardadas en: {output_img_dir}/resultados_metricas.json")
    print(f"- Gráficas exportadas en la carpeta: {output_img_dir}/")
    print("\n¡Fase 3 de Análisis y Evaluación completada exitosamente!")
    return results

if __name__ == "__main__":
    analizar_y_evaluar_modelos()
