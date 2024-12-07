# Librerías necesarias
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from scipy import optimize as sco

# Configuración de la página de Streamlit
st.set_page_config(page_title="Análisis de Portafolio y Modelo Black-Litterman", layout="wide")
st.sidebar.title("Análisis de Portafolio y Optimización Black-Litterman")

# Entrada de activos, pesos iniciales y benchmark
simbolos_input = st.sidebar.text_input("Ingrese los símbolos de los activos separados por comas (e.g., EMB, EWW, GNR):", "EMB, EWW, GNR, RXI, SHY")
benchmark_input = st.sidebar.text_input("Ingrese el símbolo del benchmark (e.g., ^GSPC):", "^GSPC")
simbolos = [s.strip() for s in simbolos_input.split(',')]
benchmark = benchmark_input.strip()

# Selección de ventana de tiempo
end_date = pd.Timestamp.today()
start_date = st.sidebar.date_input("Fecha inicial de análisis", value=(end_date - pd.DateOffset(years=1)))
if pd.Timestamp(start_date) >= end_date:
    st.sidebar.error("La fecha inicial debe ser anterior a la fecha final.")
    st.stop()

# Descargar datos de precios históricos
try:
    precios = yf.download(simbolos + [benchmark], start=pd.Timestamp(start_date), end=end_date)['Adj Close']
    precios = precios.dropna()
except Exception as e:
    st.error(f"Error al descargar los datos: {e}")
    st.stop()

# Calcular rendimientos diarios
rendimientos = precios.pct_change().dropna()
media_rendimientos = rendimientos.mean() * 252
covarianza = rendimientos.cov() * 252

# Asegurar que los índices de los activos coincidan
activos_validos = [activo for activo in simbolos if activo in media_rendimientos.index and activo in covarianza.index]
media_rendimientos = media_rendimientos[activos_validos]
volatilidades = np.sqrt(np.diag(covarianza.loc[activos_validos, activos_validos]))

# Crear DataFrame con métricas básicas
st.header("Análisis Tradicional del Portafolio")
st.write("Rendimientos promedio y volatilidades históricas (anualizadas):")
metricas_basicas = pd.DataFrame({
    "Activo": activos_validos,
    "Rendimiento Promedio (%)": media_rendimientos * 100,
    "Volatilidad (%)": volatilidades * 100
}).set_index("Activo")
st.dataframe(metricas_basicas.style.format("{:.2f}"))

# Black-Litterman: Incorporar vistas
st.sidebar.title("Vistas para el Modelo Black-Litterman")
vistas = {}
for simbolo in activos_validos:
    vistas[simbolo] = st.sidebar.number_input(f"Rendimiento esperado para {simbolo} (%):", value=media_rendimientos[simbolo] * 100)

vistas_vector = np.array([vistas[s] / 100 for s in activos_validos])
P = np.eye(len(activos_validos))  # Supongamos una matriz de identidad para views específicas

# Parámetros de incertidumbre
tau = st.sidebar.number_input("Valor de tau (incertidumbre en el mercado):", value=0.025, min_value=0.001, max_value=1.0, step=0.001)
omega = np.diag(np.diag(tau * covarianza.loc[activos_validos, activos_validos]))  # Proporcional a tau * covarianza

# Modelo Black-Litterman
prior = covarianza.loc[activos_validos, activos_validos] @ np.ones(len(activos_validos)) / len(activos_validos)
M_inverse = np.linalg.inv(tau * covarianza.loc[activos_validos, activos_validos])
omega_inverse = np.linalg.inv(omega)
combined_mean = np.linalg.inv(M_inverse + P.T @ omega_inverse @ P) @ (M_inverse @ prior + P.T @ omega_inverse @ vistas_vector)
combined_cov = np.linalg.inv(M_inverse + P.T @ omega_inverse @ P)

# Optimización de portafolio con las nuevas expectativas
def calcular_rendimiento(pesos):
    return np.dot(pesos, combined_mean)

def calcular_volatilidad(pesos):
    return np.sqrt(np.dot(pesos.T, np.dot(combined_cov, pesos)))

def sharpe_ratio(pesos, rf=0.02):
    rendimiento = calcular_rendimiento(pesos)
    volatilidad = calcular_volatilidad(pesos)
    return (rendimiento - rf) / volatilidad

# Restricciones y optimización
restricciones = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Pesos deben sumar 1
limites = tuple((0, 1) for _ in range(len(activos_validos)))  # Pesos entre 0 y 1
resultado = sco.minimize(lambda x: -sharpe_ratio(x), np.ones(len(activos_validos)) / len(activos_validos), method="SLSQP", bounds=limites, constraints=restricciones)

# Resultados
pesos_optimizados = resultado.x
rendimiento_optimo = calcular_rendimiento(pesos_optimizados)
volatilidad_optima = calcular_volatilidad(pesos_optimizados)
sharpe_optimo = sharpe_ratio(pesos_optimizados)

# Mostrar resultados
st.header("Optimización Black-Litterman")
col1, col2, col3 = st.columns(3)
col1.metric("Rendimiento Optimizado", f"{rendimiento_optimo:.2%}")
col2.metric("Volatilidad Optimizada", f"{volatilidad_optima:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_optimo:.2f}")

# Pesos del portafolio
st.subheader("Pesos del Portafolio")
pesos_df = pd.DataFrame({
    "Activo": activos_validos,
    "Peso Inicial": np.ones(len(activos_validos)) / len(activos_validos),
    "Peso Optimizado": pesos_optimizados
})

# Aplicar formato solo a columnas numéricas
st.dataframe(
    pesos_df.style.format(subset=["Peso Inicial", "Peso Optimizado"], formatter="{:.4f}")
)

# Gráficos
st.subheader("Comparación de Pesos Iniciales vs Optimizados")
pesos_chart = pd.DataFrame({
    "Activo": activos_validos,
    "Pesos Iniciales": np.ones(len(activos_validos)) / len(activos_validos),
    "Pesos Optimizados": pesos_optimizados
}).set_index("Activo")
st.bar_chart(pesos_chart)
