import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# CONFIGURACIÓN GENERAL
# -------------------------------------------------
st.set_page_config(
    page_title="Forecasting de Ventas",
    page_icon="📈",
    layout="wide"
)


# -------------------------------------------------
# LOGIN
# -------------------------------------------------
def login():
    st.markdown("""
    <style>
    .login-title {
        font-size: 2.4rem;
        font-weight: 850;
        color: #0F172A;
        text-align: center;
        margin-top: 4rem;
    }
    .login-subtitle {
        text-align: center;
        color: #64748B;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-title">Acceso al Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Ingrese sus credenciales para continuar</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])

    with col2:
        usuario = st.text_input("ID")
        password = st.text_input("Contraseña", type="password")

        if st.button("Ingresar", use_container_width=True):
            if usuario == "admin" and password == "123":
                st.session_state["logueado"] = True
                st.rerun()
            else:
                st.error("ID o contraseña incorrectos")


if "logueado" not in st.session_state:
    st.session_state["logueado"] = False

if not st.session_state["logueado"]:
    login()
    st.stop()


# -------------------------------------------------
# AUTOACTUALIZACIÓN
# -------------------------------------------------
REFRESH_SECONDS = 60

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh >= REFRESH_SECONDS:
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()


# -------------------------------------------------
# ESTILOS PREMIUM
# -------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #EEF2F7 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.titulo-principal {
    font-size: 2.7rem;
    font-weight: 850;
    color: #0F172A;
    margin-bottom: 0.2rem;
    letter-spacing: -0.03em;
}

.subtitulo {
    font-size: 1.05rem;
    color: #64748B;
    margin-bottom: 2rem;
}

.card {
    background: white;
    padding: 1.35rem;
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    border: 1px solid #E2E8F0;
    min-height: 120px;
}

.kpi-title {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-value {
    font-size: 2rem;
    color: #0F172A;
    font-weight: 850;
    margin-top: 0.4rem;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0F172A;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.chart-card {
    background: white;
    padding: 1.2rem;
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    border: 1px solid #E2E8F0;
    margin-bottom: 1.2rem;
}

section[data-testid="stSidebar"] {
    background: #0F172A;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.stButton button {
    border-radius: 12px;
    background: #2563EB;
    color: white;
    border: none;
    font-weight: 700;
}

.stDownloadButton button {
    border-radius: 12px;
    background: #0F172A;
    color: white;
    border: none;
    font-weight: 700;
}

.stAlert {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# TÍTULO
# -------------------------------------------------
st.markdown(
    '<div class="titulo-principal">Dashboard de Predicción de Ventas</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">Panel ejecutivo con análisis histórico, métricas clave, selector de familia y forecast automático conectado a Google Sheets.</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# FUNCIONES
# -------------------------------------------------
@st.cache_data(ttl=60)
def cargar_datos_google_sheets(url):
    return pd.read_csv(url)


def preparar_datos(df):
    df = df.copy()

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["ventas_totales"] = pd.to_numeric(df["ventas_totales"], errors="coerce")

    df = df.dropna(subset=["FECHA", "ventas_totales"])
    df = df.sort_values("FECHA").reset_index(drop=True)

    df["year"] = df["FECHA"].dt.year
    df["month"] = df["FECHA"].dt.month
    df["month_name"] = df["FECHA"].dt.month_name()
    df["day_of_week"] = df["FECHA"].dt.dayofweek
    df["dia_nombre"] = df["FECHA"].dt.day_name()
    df["fin_semana"] = (df["day_of_week"] >= 5).astype(int)

    df["lag_1"] = df["ventas_totales"].shift(1)
    df["lag_7"] = df["ventas_totales"].shift(7)
    df["lag_14"] = df["ventas_totales"].shift(14)
    df["lag_30"] = df["ventas_totales"].shift(30)

    df["media_7"] = df["ventas_totales"].rolling(7).mean()
    df["media_30"] = df["ventas_totales"].rolling(30).mean()

    return df


def entrenar_mejor_modelo(df):
    df_model = df.dropna().copy()

    features = [
        "year", "month", "day_of_week", "fin_semana",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "media_7", "media_30"
    ]

    X = df_model[features]
    y = df_model["ventas_totales"]

    split = int(len(df_model) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    modelos = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    resultados = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)

        resultados.append({
            "modelo": nombre,
            "objeto_modelo": modelo,
            "pred": pred,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    resultados_df = pd.DataFrame(resultados)

    resultados_df["rank_RMSE"] = resultados_df["RMSE"].rank(method="min", ascending=True)
    resultados_df["rank_MAE"] = resultados_df["MAE"].rank(method="min", ascending=True)
    resultados_df["rank_R2"] = resultados_df["R2"].rank(method="min", ascending=False)

    resultados_df["score_final"] = (
        resultados_df["rank_RMSE"] +
        resultados_df["rank_MAE"] +
        resultados_df["rank_R2"]
    )

    mejor = resultados_df.sort_values("score_final").iloc[0]

    return (
        mejor["objeto_modelo"],
        mejor["modelo"],
        features,
        split,
        y_train,
        y_test,
        mejor["pred"],
        mejor["MAE"],
        mejor["MSE"],
        mejor["RMSE"],
        mejor["R2"],
        df_model,
        resultados_df.drop(columns=["objeto_modelo", "pred"])
    )


def predecir_30_dias(df_model, modelo, features, dias=30):
    df_future = df_model.copy()
    predicciones = []

    for _ in range(dias):
        ultima_fecha = df_future["FECHA"].iloc[-1]
        nueva_fecha = ultima_fecha + pd.Timedelta(days=1)

        nueva_fila = {
            "FECHA": nueva_fecha,
            "ventas_totales": np.nan,
            "year": nueva_fecha.year,
            "month": nueva_fecha.month,
            "month_name": nueva_fecha.month_name(),
            "day_of_week": nueva_fecha.dayofweek,
            "dia_nombre": nueva_fecha.day_name(),
            "fin_semana": 1 if nueva_fecha.dayofweek >= 5 else 0,
            "lag_1": df_future["ventas_totales"].iloc[-1],
            "lag_7": df_future["ventas_totales"].iloc[-7],
            "lag_14": df_future["ventas_totales"].iloc[-14],
            "lag_30": df_future["ventas_totales"].iloc[-30],
            "media_7": df_future["ventas_totales"].tail(7).mean(),
            "media_30": df_future["ventas_totales"].tail(30).mean()
        }

        X_new = pd.DataFrame([nueva_fila])[features]
        pred = modelo.predict(X_new)[0]

        if pred < 0:
            pred = 0

        nueva_fila["ventas_totales"] = pred

        predicciones.append({
            "FECHA": nueva_fecha,
            "ventas_predichas": round(pred, 2)
        })

        df_future = pd.concat(
            [df_future, pd.DataFrame([nueva_fila])],
            ignore_index=True
        )

    return pd.DataFrame(predicciones)


def card_kpi(titulo, valor):
    st.markdown(f"""
    <div class="card">
        <div class="kpi-title">{titulo}</div>
        <div class="kpi-value">{valor}</div>
    </div>
    """, unsafe_allow_html=True)


def chart_container(fig):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def seleccionar_seccion(nombre):
    st.session_state["seccion"] = nombre


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Datos en la nube")
st.sidebar.success("Conectado a Google Sheets")
st.sidebar.info("Actualización automática cada 60 segundos")

if st.sidebar.button("Actualizar ahora"):
    st.session_state.last_refresh = time.time()
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("Cerrar sesión"):
    st.session_state["logueado"] = False
    st.rerun()

st.sidebar.caption(
    f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
)


# -------------------------------------------------
# CONEXIÓN A GOOGLE SHEETS
# -------------------------------------------------
url = "https://docs.google.com/spreadsheets/d/1VLTbAFyw6XYbQLKj32bAg6Bn3qHDf21nGty_uoTVY7o/gviz/tq?tqx=out:csv&gid=355174621"

try:
    df = cargar_datos_google_sheets(url)

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_")
    )

    if "FECHA" not in df.columns:
        st.error("No se encontró la columna 'FECHA'.")
        st.write("Columnas detectadas:", df.columns.tolist())
        st.dataframe(df.head())
        st.stop()

    if "FAMILIA" not in df.columns:
        st.error("No se encontró la columna 'FAMILIA'.")
        st.write("Columnas detectadas:", df.columns.tolist())
        st.dataframe(df.head())
        st.stop()

    if "VENTAS_TOTALES" in df.columns:
        df = df.rename(columns={"VENTAS_TOTALES": "ventas_totales"})
    elif "TOTAL" in df.columns:
        df = df.rename(columns={"TOTAL": "ventas_totales"})
    elif "VENTA" in df.columns:
        df = df.rename(columns={"VENTA": "ventas_totales"})
    else:
        st.error("No se encontró columna de ventas. Debe llamarse 'ventas_totales', 'TOTAL' o 'VENTA'.")
        st.write("Columnas detectadas:", df.columns.tolist())
        st.dataframe(df.head())
        st.stop()

except Exception as e:
    st.error("No se pudo conectar correctamente con Google Sheets.")
    st.write(e)
    st.stop()


# -------------------------------------------------
# SELECTOR DE FAMILIA
# -------------------------------------------------
familias = sorted(df["FAMILIA"].dropna().astype(str).unique())

familia_seleccionada = st.sidebar.selectbox(
    "Seleccionar familia",
    familias
)

df = df[df["FAMILIA"].astype(str) == familia_seleccionada].copy()


# -------------------------------------------------
# PROCESAMIENTO
# -------------------------------------------------
df = preparar_datos(df)

if len(df.dropna()) < 31:
    st.error("Se necesitan al menos 31 registros válidos para generar lags y entrenar el modelo.")
    st.stop()


(
    modelo,
    nombre_modelo,
    features,
    split,
    y_train,
    y_test,
    pred,
    mae,
    mse,
    rmse,
    r2,
    df_model,
    resultados_modelos
) = entrenar_mejor_modelo(df)

forecast_30 = predecir_30_dias(df_model, modelo, features, dias=30)


# -------------------------------------------------
# MENÚ DE BOTONES
# -------------------------------------------------
if "seccion" not in st.session_state:
    st.session_state["seccion"] = "inicio"

st.markdown(
    f'<div class="section-title">Familia seleccionada: {familia_seleccionada}</div>',
    unsafe_allow_html=True
)

b1, b2, b3, b4 = st.columns(4)

with b1:
    if st.button("Resumen ejecutivo", use_container_width=True):
        seleccionar_seccion("resumen")

with b2:
    if st.button("Promedio por día", use_container_width=True):
        seleccionar_seccion("promedio_dia")

with b3:
    if st.button("Porcentaje por día", use_container_width=True):
        seleccionar_seccion("porcentaje_dia")

with b4:
    if st.button("Promedio por mes", use_container_width=True):
        seleccionar_seccion("promedio_mes")

b5, b6, b7, b8 = st.columns(4)

with b5:
    if st.button("Crecimiento anual", use_container_width=True):
        seleccionar_seccion("crecimiento_anual")

with b6:
    if st.button("Serie histórica", use_container_width=True):
        seleccionar_seccion("serie_historica")

with b7:
    if st.button("Modelo predictivo", use_container_width=True):
        seleccionar_seccion("modelo")

with b8:
    if st.button("Forecast 30 días", use_container_width=True):
        seleccionar_seccion("forecast")


# -------------------------------------------------
# DATAFRAMES AUXILIARES
# -------------------------------------------------
orden_dias = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

ventas_dia = (
    df.groupby("dia_nombre", as_index=False)["ventas_totales"]
    .mean()
    .rename(columns={"ventas_totales": "promedio_ventas"})
)

ventas_dia["dia_nombre"] = pd.Categorical(
    ventas_dia["dia_nombre"],
    categories=orden_dias,
    ordered=True
)

ventas_dia = ventas_dia.sort_values("dia_nombre")

pct_dia = (
    df.groupby("dia_nombre", as_index=False)["ventas_totales"]
    .sum()
)

pct_dia["porcentaje"] = (
    pct_dia["ventas_totales"] / pct_dia["ventas_totales"].sum() * 100
)

pct_dia["dia_nombre"] = pd.Categorical(
    pct_dia["dia_nombre"],
    categories=orden_dias,
    ordered=True
)

pct_dia = pct_dia.sort_values("dia_nombre")

orden_meses = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

ventas_mes = (
    df.groupby("month_name", as_index=False)["ventas_totales"]
    .mean()
    .rename(columns={"ventas_totales": "promedio_ventas"})
)

ventas_mes["month_name"] = pd.Categorical(
    ventas_mes["month_name"],
    categories=orden_meses,
    ordered=True
)

ventas_mes = ventas_mes.sort_values("month_name")

ventas_anio = df.groupby("year", as_index=False)["ventas_totales"].sum()
ventas_anio["crecimiento_%"] = ventas_anio["ventas_totales"].pct_change() * 100


# -------------------------------------------------
# SECCIONES
# -------------------------------------------------
seccion = st.session_state["seccion"]

if seccion == "inicio":
    st.info("Seleccione un botón para visualizar una sección del dashboard.")


elif seccion == "resumen":
    st.markdown('<div class="section-title">Resumen ejecutivo</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        card_kpi("Total vendido", f"{df['ventas_totales'].sum():,.2f}")

    with k2:
        card_kpi("Promedio diario", f"{df['ventas_totales'].mean():,.2f}")

    with k3:
        card_kpi("Máximo diario", f"{df['ventas_totales'].max():,.2f}")

    with k4:
        card_kpi("Mínimo diario", f"{df['ventas_totales'].min():,.2f}")

    with k5:
        card_kpi("Última fecha", str(df["FECHA"].max().date()))


elif seccion == "promedio_dia":
    st.markdown('<div class="section-title">Promedio de ventas por día</div>', unsafe_allow_html=True)

    fig1 = px.bar(
        ventas_dia,
        x="dia_nombre",
        y="promedio_ventas",
        title=f"Promedio de ventas por día - {familia_seleccionada}",
        labels={"dia_nombre": "Día", "promedio_ventas": "Promedio de ventas"},
        text_auto=".2s"
    )

    fig1.update_layout(
        template="plotly_white",
        height=500,
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig1)


elif seccion == "porcentaje_dia":
    st.markdown('<div class="section-title">Participación porcentual por día</div>', unsafe_allow_html=True)

    fig2 = px.pie(
        pct_dia,
        names="dia_nombre",
        values="porcentaje",
        title=f"Participación porcentual por día - {familia_seleccionada}",
        hole=0.45
    )

    fig2.update_layout(
        template="plotly_white",
        height=500,
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig2)


elif seccion == "promedio_mes":
    st.markdown('<div class="section-title">Promedio de ventas por mes</div>', unsafe_allow_html=True)

    fig3 = px.bar(
        ventas_mes,
        x="month_name",
        y="promedio_ventas",
        title=f"Promedio de ventas por mes - {familia_seleccionada}",
        labels={"month_name": "Mes", "promedio_ventas": "Promedio de ventas"},
        text_auto=".2s"
    )

    fig3.update_layout(
        template="plotly_white",
        height=500,
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig3)


elif seccion == "crecimiento_anual":
    st.markdown('<div class="section-title">Crecimiento histórico por año</div>', unsafe_allow_html=True)

    fig4 = px.line(
        ventas_anio,
        x="year",
        y="ventas_totales",
        title=f"Crecimiento histórico por año - {familia_seleccionada}",
        markers=True,
        labels={"year": "Año", "ventas_totales": "Ventas totales"}
    )

    fig4.update_layout(
        template="plotly_white",
        height=500,
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig4)

    st.dataframe(ventas_anio, use_container_width=True)


elif seccion == "serie_historica":
    st.markdown('<div class="section-title">Serie histórica de ventas</div>', unsafe_allow_html=True)

    fig5 = px.line(
        df,
        x="FECHA",
        y="ventas_totales",
        title=f"Comportamiento histórico de ventas - {familia_seleccionada}",
        labels={"FECHA": "Fecha", "ventas_totales": "Ventas"}
    )

    fig5.update_layout(
        template="plotly_white",
        height=500,
        title_font_size=20,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig5)


elif seccion == "modelo":
    st.markdown('<div class="section-title">Resumen del modelo predictivo</div>', unsafe_allow_html=True)

    st.success(
        f"Mejor modelo seleccionado para {familia_seleccionada}: {nombre_modelo} "
        f"| MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R²: {r2:.3f}"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        card_kpi("Modelo ganador", nombre_modelo)

    with m2:
        card_kpi("MAE", f"{mae:,.2f}")

    with m3:
        card_kpi("RMSE", f"{rmse:,.2f}")

    with m4:
        card_kpi("R²", f"{r2:.3f}")

    st.markdown('<div class="section-title">Ranking de modelos</div>', unsafe_allow_html=True)

    st.dataframe(
        resultados_modelos.sort_values("score_final"),
        use_container_width=True
    )

    st.markdown('<div class="section-title">Comparación real vs predicción</div>', unsafe_allow_html=True)

    fechas_train = df_model.iloc[:split]["FECHA"]
    fechas_test = df_model.iloc[split:]["FECHA"]

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=fechas_train,
        y=y_train.values,
        mode="lines",
        name="Train real"
    ))

    fig6.add_trace(go.Scatter(
        x=fechas_test,
        y=y_test.values,
        mode="lines",
        name="Test real"
    ))

    fig6.add_trace(go.Scatter(
        x=fechas_test,
        y=pred,
        mode="lines",
        name="Predicción"
    ))

    fig6.update_layout(
        template="plotly_white",
        title=f"Ventas reales vs predicción del modelo - {familia_seleccionada}",
        height=500,
        title_font_size=20,
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig6)


elif seccion == "forecast":
    st.markdown('<div class="section-title">Proyección de los próximos 30 días</div>', unsafe_allow_html=True)

    fig7 = go.Figure()

    fig7.add_trace(go.Scatter(
        x=df["FECHA"].tail(90),
        y=df["ventas_totales"].tail(90),
        mode="lines",
        name="Histórico reciente"
    ))

    fig7.add_trace(go.Scatter(
        x=forecast_30["FECHA"],
        y=forecast_30["ventas_predichas"],
        mode="lines+markers",
        name="Forecast 30 días"
    ))

    fig7.update_layout(
        template="plotly_white",
        title=f"Pronóstico de ventas a 30 días - {familia_seleccionada}",
        height=500,
        title_font_size=20,
        xaxis_title="Fecha",
        yaxis_title="Ventas",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    chart_container(fig7)

    st.markdown('<div class="section-title">Tabla de pronóstico</div>', unsafe_allow_html=True)

    st.dataframe(forecast_30, use_container_width=True)

    csv = forecast_30.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar forecast en CSV",
        data=csv,
        file_name=f"forecast_30_dias_{familia_seleccionada}.csv",
        mime="text/csv"
    )
