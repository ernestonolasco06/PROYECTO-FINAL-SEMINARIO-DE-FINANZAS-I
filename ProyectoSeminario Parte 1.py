
"""
Proyecto Seminario de Finanzas (INCISO 1)
Integrantes:
-Luis Daniel COntreras Hernández
-Ernesto Nolasco Cruz Salvador
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from numpy.linalg import multi_dot
from scipy.stats import kurtosis, skew, norm
from scipy import optimize as sco
from datetime import datetime, timedelta
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Configuración de la página de Streamlit
st.set_page_config(page_title="Analizador de Portafolio", layout="wide")
st.sidebar.title("Analizador de Portafolio de Inversión")

# Entrada de símbolos y pesos
simbolos_input = st.sidebar.text_input("Ingrese los símbolos de las acciones separados por comas como EMB, EWW, etc:", "EMB, EWW, GNR, RXI, SHY")
pesos_input = st.sidebar.text_input("Ingrese los pesos correspondientes separados por comas (deben sumar 1):", "0.2,0.2,0.2,0.2,0.2")

simbolos = [s.strip() for s in simbolos_input.split(',')]
pesos = [float(w.strip()) for w in pesos_input.split(',')]

# Validar los pesos
if len(simbolos) != len(pesos) or abs(sum(pesos) - 1) > 1e-6:
    st.sidebar.error("El número de símbolos debe coincidir con el número de pesos, y los pesos deben sumar 1.")
    st.stop()

# Selección del benchmark
benchmark_options = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "ACWI": "ACWI"
}
selected_benchmark = st.sidebar.selectbox("Seleccione el benchmark:", list(benchmark_options.keys()))
benchmark = benchmark_options[selected_benchmark]

# Selección de la ventana de tiempo
end_date = datetime.now()
start_date_options = {
    "1 mes": end_date - timedelta(days=30),
    "3 meses": end_date - timedelta(days=90),
    "6 meses": end_date - timedelta(days=180),
    "1 año": end_date - timedelta(days=365),
    "3 años": end_date - timedelta(days=3*365),
    "5 años": end_date - timedelta(days=5*365),
    "10 años": end_date - timedelta(days=10*365)
}
selected_window = st.sidebar.selectbox("Seleccione la ventana de tiempo para el análisis:", list(start_date_options.keys()))
start_date = start_date_options[selected_window]

# Descargar precios históricos
all_symbols = simbolos + [benchmark]
asset_prices = yf.download(all_symbols, start=start_date, end=end_date)['Close'].dropna()

# Calcular métricas de rendimiento
returns = asset_prices[simbolos].pct_change().dropna()
cumulative_returns = (1 + returns).cumprod() - 1
normalized_prices = asset_prices / asset_prices.iloc[0] * 100

# Funciones auxiliares para cálculo de métricas

def calcular_rendimientos_portafolio(returns, weights):
    return (returns * weights).sum(axis=1)

def calcular_beta(asset_returns, market_returns):
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else np.nan

def calcular_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calcular_sortino_ratio(returns, risk_free_rate=0.02, target_return=0):
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * excess_returns.mean() / downside_deviation if downside_deviation != 0 else np.nan

def calcular_var_cvar(returns, confidence=0.95):
    VaR = returns.quantile(1 - confidence)
    CVaR = returns[returns <= VaR].mean()
    return VaR, CVaR

def calcular_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# Calcular rendimientos del portafolio
portfolio_returns = calcular_rendimientos_portafolio(returns, pesos)
portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1

# Crear pestañas para el análisis
tab1, tab2, tab3 = st.tabs(["Análisis de Activos Individuales", "Análisis del Portafolio", "Simulación y Optimización"])

# Análisis de Activos Individuales
with tab1:
    st.header("Proyecto de Seminario de Finanzas")
    st.header("Análisis de Activos Individuales")
    st.subheader("Integrantes:")
    st.subheader("Contreras Hernández Luis Daniel")
    st.subheader("Cruz Salvador Ernesto Nolasco")
    selected_asset = st.selectbox("Seleccione un activo para analizar:", simbolos)
    var_95, cvar_95 = calcular_var_cvar(returns[selected_asset])
    drawdown = calcular_drawdown(returns[selected_asset])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Total", f"{cumulative_returns[selected_asset].iloc[-1]:.2%}")
    col2.metric("Sharpe Ratio", f"{calcular_sharpe_ratio(returns[selected_asset]):.2f}")
    col3.metric("Sortino Ratio", f"{calcular_sortino_ratio(returns[selected_asset]):.2f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("VaR 95%", f"{var_95:.2%}")
    col5.metric("CVaR 95%", f"{cvar_95:.2%}")
    col6.metric("Drawdown", f"{drawdown:.2%}")
    
    # Gráfico de precio normalizado
    fig_asset = go.Figure()
    fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[selected_asset], name=selected_asset))
    if benchmark in asset_prices.columns:
        fig_asset.add_trace(go.Scatter(x=normalized_prices.index, y=normalized_prices[benchmark], name=selected_benchmark))
    fig_asset.update_layout(title=f'Precio Normalizado: {selected_asset} vs {selected_benchmark} (Base 100)', xaxis_title='Fecha', yaxis_title='Precio Normalizado')
    st.plotly_chart(fig_asset, use_container_width=True)

    # Tabla resumen de estadísticas individuales
    stats_data = {
        'media': returns.mean(),
        'sesgo': returns.skew(),
        'exceso de curtosis': returns.kurtosis(),
        'VaR': returns.apply(lambda x: calcular_var_cvar(x)[0]),
        'CVaR': returns.apply(lambda x: calcular_var_cvar(x)[1]),
        'Sharpe ratio': returns.apply(calcular_sharpe_ratio),
        'Sortino': returns.apply(calcular_sortino_ratio),
        'Drawdown': returns.apply(calcular_drawdown)
    }
    stats_df = pd.DataFrame(stats_data)
    st.subheader("Estadísticas de los Activos")
    st.dataframe(stats_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn"))

# Análisis del Portafolio
with tab2:
    st.header("Análisis del Portafolio")
    portfolio_var_95, portfolio_cvar_95 = calcular_var_cvar(portfolio_returns)
    portfolio_drawdown = calcular_drawdown(portfolio_returns)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Total del Portafolio", f"{portfolio_cumulative_returns.iloc[-1]:.2%}")
    col2.metric("Sharpe Ratio del Portafolio", f"{calcular_sharpe_ratio(portfolio_returns):.2f}")
    col3.metric("Sortino Ratio del Portafolio", f"{calcular_sortino_ratio(portfolio_returns):.2f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("VaR 95% del Portafolio", f"{portfolio_var_95:.2%}")
    col5.metric("CVaR 95% del Portafolio", f"{portfolio_cvar_95:.2%}")
    col6.metric("Drawdown del Portafolio", f"{portfolio_drawdown:.2%}")

    # Gráfico de rendimientos acumulados del portafolio vs benchmark
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(x=portfolio_cumulative_returns.index, y=portfolio_cumulative_returns, name='Portafolio'))
    if benchmark in asset_prices.columns:
        benchmark_returns = asset_prices[benchmark].pct_change().dropna()
        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
        fig_cumulative.add_trace(go.Scatter(x=benchmark_cumulative_returns.index, y=benchmark_cumulative_returns, name=selected_benchmark))
    fig_cumulative.update_layout(title=f'Rendimientos Acumulados: Portafolio vs {selected_benchmark}', xaxis_title='Fecha', yaxis_title='Rendimiento Acumulado')
    st.plotly_chart(fig_cumulative, use_container_width=True)

# Simulación y Optimización de Portafolios
with tab3:
    st.header("Simulación y Optimización de Portafolios")
    historic_data = asset_prices.loc[:start_date + pd.DateOffset(years=10)]

    # Normalización de precios
    normalized_data = historic_data / historic_data.iloc[0]
    fig = px.line(normalized_data, title='Precios estandarizados')
    st.plotly_chart(fig, use_container_width=True)

    # Simulación de Portafolios
    def portfolio_simulation(returns):
        numofasset = len(returns.columns)
        rets = []
        vols = []
        wts = []
        for _ in range(5000):
            weights = np.random.random(numofasset)
            weights /= sum(weights)
            rets.append(np.dot(weights, returns.mean() * 252))
            vols.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))
            wts.append(weights)
        portdf = 100 * pd.DataFrame({
            'port_rets': np.array(rets).flatten(),
            'port_vols': np.array(vols).flatten(),
            'weights': list(np.array(wts))
        })
        portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']
        portdf.index.name = 'sim_id'
        return round(portdf, 2)

    simulated_portfolios = portfolio_simulation(returns)

    # Visualización de los portafolios simulados
    fig = px.scatter(
        simulated_portfolios,
        x="port_vols",
        y="port_rets",
        color="sharpe_ratio",
        labels={"port_vols": "Expected Volatility", "port_rets": "Expected Return", "sharpe_ratio": "Sharpe Ratio"},
        title="Monte Carlo Simulated Portfolio"
    ).update_traces(mode="markers", marker=dict(symbol="cross"))
    st.plotly_chart(fig, use_container_width=True)

    # Encontrar el portafolio con el Max Sharpe Ratio
    max_sharpe_ratio_index = simulated_portfolios["sharpe_ratio"].idxmax()
    msrpwts = simulated_portfolios["weights"][max_sharpe_ratio_index]
    st.write("Pesos del Portafolio de Máximo Sharpe Ratio:")
    st.write(dict(zip(simbolos, np.around(msrpwts, 2))))

    # Optimización para minimizar la volatilidad
    def portfolio_variance(weights):
        return portfolio_stats(weights)[1] ** 2

    def portfolio_stats(weights):
        weights = np.array(weights)
        port_rets = np.dot(weights, returns.mean() * 252)
        port_vols = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return np.array([port_rets, port_vols, port_rets / port_vols]).flatten()

    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(len(simbolos)))
    initial_wts = len(simbolos) * [1. / len(simbolos)]
    opt_var = sco.minimize(portfolio_variance, initial_wts, method="SLSQP", bounds=bnds, constraints=cons)

    # Pesos de portafolio con mínimo de volatilidad
    st.write("Pesos del Portafolio de Mínima Volatilidad:")
    st.write(dict(zip(simbolos, np.around(opt_var['x'], 2))))

    # Optimización para maximizar el Sharpe Ratio
    def min_sharpe_ratio(weights):
        return -portfolio_stats(weights)[2]

    opt_sharpe = sco.minimize(min_sharpe_ratio, initial_wts, method="SLSQP", bounds=bnds, constraints=cons)

    # Pesos del portafolio de máximo Sharpe Ratio
    st.write("Pesos del Portafolio de Máximo Sharpe Ratio (Optimización):")
    st.write(dict(zip(simbolos, np.around(opt_sharpe['x'], 2))))

    # Optimización con rendimiento objetivo del 10%
    def portfolio_mean_return(weights):
        return np.dot(weights, returns.mean() * 252)

    desired_return = 0.10  # 10%
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_mean_return(x) - desired_return}
    ]
    portfolio_opt = sco.minimize(portfolio_variance, initial_wts, method='SLSQP', bounds=bnds, constraints=constraints)

    # Pesos del portafolio con mínimo de volatilidad y objetivo de rendimiento
    st.write("Pesos del Portafolio de Mínima Volatilidad con Objetivo de Rendimiento del 10% Anual:")
    st.write(dict(zip(simbolos, np.around(portfolio_opt['x'], 2))))
