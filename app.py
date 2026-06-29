# app.py
# ============================================================
# MARKET DONNA | Dashboard Ejecutivo de Predicción de Pedidos
# Diseño profesional + Supabase + Machine Learning
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
# from reportlab.lib.enums import TA_CENTER
# from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Market Donna | Forecast",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILOS PREMIUM
# ============================================================

st.markdown("""
<style>
/* =========================
   FUENTE Y FONDO GENERAL
========================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 26%),
        linear-gradient(135deg, #F8FAFC 0%, #EEF2F7 100%);
}

/* =========================
   OCULTAR ELEMENTOS STREAMLIT
========================= */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* =========================
   CONTENEDORES
========================= */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.glass-panel {
    background: rgba(255,255,255,0.78);
    border: 1px solid rgba(226,232,240,0.9);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(16px);
}

.hero {
    background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.92)),
        radial-gradient(circle at top right, rgba(34,197,94,0.22), transparent 36%);
    border-radius: 30px;
    padding: 34px 38px;
    color: white;
    box-shadow: 0 28px 65px rgba(15, 23, 42, 0.22);
    margin-bottom: 24px;
}

.hero-title {
    font-size: 42px;
    font-weight: 850;
    letter-spacing: -1.2px;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #CBD5E1;
    font-size: 17px;
    line-height: 1.6;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    color: #E0F2FE;
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 16px;
}

/* =========================
   LOGIN
========================= */
.login-wrapper {
    max-width: 470px;
    margin: 7vh auto 0 auto;
}

.login-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 32px;
    padding: 34px;
    box-shadow: 0 30px 90px rgba(15, 23, 42, 0.13);
    backdrop-filter: blur(18px);
}

.login-logo {
    width: 68px;
    height: 68px;
    border-radius: 22px;
    background: linear-gradient(135deg, #1D4ED8, #10B981);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 34px;
    margin-bottom: 18px;
    box-shadow: 0 18px 35px rgba(37,99,235,0.24);
}

.login-title {
    font-size: 34px;
    font-weight: 850;
    color: #0F172A;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.login-subtitle {
    color: #64748B;
    font-size: 15.5px;
    margin-bottom: 22px;
}

/* =========================
   MÉTRICAS
========================= */
.metric-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(226,232,240,0.95);
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 16px 38px rgba(15,23,42,0.07);
    transition: all 0.18s ease-in-out;
    min-height: 142px;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 22px 48px rgba(15,23,42,0.10);
}

.metric-icon {
    width: 44px;
    height: 44px;
    border-radius: 16px;
    background: #EFF6FF;
    color: #1D4ED8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 14px;
}

.metric-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .45px;
}

.metric-value {
    color: #0F172A;
    font-size: 30px;
    font-weight: 850;
    letter-spacing: -0.7px;
    margin-top: 6px;
}

/* =========================
   SECCIONES
========================= */
.section-title {
    color: #0F172A;
    font-size: 24px;
    font-weight: 850;
    letter-spacing: -0.6px;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #64748B;
    font-size: 14.5px;
    margin-bottom: 18px;
}

.soft-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(226,232,240,0.96);
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 16px 42px rgba(15, 23, 42, 0.06);
    margin-bottom: 20px;
}

.user-chip {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 18px;
    padding: 14px 16px;
    color: white;
    font-weight: 700;
}

.info-pill {
    display: inline-flex;
    padding: 9px 13px;
    border-radius: 999px;
    background: #EFF6FF;
    color: #1D4ED8;
    font-weight: 750;
    font-size: 13px;
    border: 1px solid #DBEAFE;
}

/* =========================
   BOTONES
========================= */
.stButton > button {
    border: 0 !important;
    border-radius: 16px !important;
    height: 50px !important;
    font-weight: 800 !important;
    color: white !important;
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    box-shadow: 0 14px 28px rgba(37,99,235,0.24) !important;
    transition: all 0.18s ease-in-out !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 36px rgba(37,99,235,0.30) !important;
}

.stDownloadButton > button {
    border: 0 !important;
    border-radius: 16px !important;
    height: 50px !important;
    font-weight: 800 !important;
    color: white !important;
    background: linear-gradient(135deg, #059669, #10B981) !important;
    box-shadow: 0 14px 28px rgba(16,185,129,0.24) !important;
}

/* =========================
   INPUTS
========================= */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 15px !important;
    border-color: #CBD5E1 !important;
    background-color: rgba(255,255,255,0.92) !important;
}

label {
    font-weight: 750 !important;
    color: #334155 !important;
}

/* =========================
   ALERTAS
========================= */
div[data-testid="stAlert"] {
    border-radius: 18px !important;
    border: 1px solid rgba(226,232,240,0.9) !important;
}

/* =========================
   TABLAS
========================= */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #E2E8F0;
}

/* =========================
   SIDEBAR
========================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 850;
    color: white;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    color: #94A3B8;
    font-size: 13px;
    margin-bottom: 18px;
}

/* =========================
   COLOR DEL TEXTO EN INPUTS
========================= */

.stTextInput input,
.stTextInput input:focus {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 600;
}

.stTextInput input::placeholder {
    color: #9CA3AF !important;
}

div[data-baseweb="input"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)
# ============================================================
# CONEXIÓN SUPABASE
# ============================================================

SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# SESIÓN
# ============================================================

if "login" not in st.session_state:
    st.session_state.login = False

if "usuario" not in st.session_state:
    st.session_state.usuario = ""

if "sede" not in st.session_state:
    st.session_state.sede = ""

# ============================================================
# LOGIN PREMIUM
# ============================================================

if not st.session_state.login:

    st.markdown("""
    <div class="login-wrapper">
        <div class="login-card">
            <div class="login-logo">🛒</div>
            <div class="login-title">Market Donna</div>
            <div class="login-subtitle">
                Plataforma inteligente para predicción de pedidos, análisis histórico
                y soporte a decisiones de compra.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col_a, col_b, col_c = st.columns([1, 1.15, 1])
        with col_b:
            usuario = st.text_input("Usuario", placeholder="Ingrese su usuario")
            password = st.text_input("Contraseña", type="password", placeholder="Ingrese su contraseña")

            if st.button("Ingresar al sistema", use_container_width=True):

                respuesta = (
                    supabase
                    .table("usuarios")
                    .select("*")
                    .eq("usuario", usuario)
                    .execute()
                )

                if len(respuesta.data) == 0:
                    st.error("Usuario no encontrado.")
                    st.stop()

                datos = respuesta.data[0]

                if datos["password"] != password:
                    st.error("Contraseña incorrecta.")
                    st.stop()

                st.session_state.login = True
                st.session_state.usuario = datos["usuario"]
                st.session_state.sede = str(datos["sede"]).upper().strip()

                st.rerun()

    st.stop()

# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(ttl=120)
def cargar_ventas():

    todos_los_datos = []
    rango_inicio = 0
    tamaño_bloque = 1000

    while True:
        respuesta = (
            supabase
            .table("ventas")
            .select("*")
            .range(rango_inicio, rango_inicio + tamaño_bloque - 1)
            .execute()
        )

        datos = respuesta.data

        if not datos:
            break

        todos_los_datos.extend(datos)

        if len(datos) < tamaño_bloque:
            break

        rango_inicio += tamaño_bloque

    df = pd.DataFrame(todos_los_datos)

    if df.empty:
        return df

    df.columns = df.columns.str.upper()

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce")
    df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")

    df["SEDE"] = df["SEDE"].astype(str).str.upper().str.strip()
    df["FAMILIA"] = df["FAMILIA"].astype(str).str.upper().str.strip()
    df["DESCRIPCIO"] = df["DESCRIPCIO"].astype(str).str.upper().str.strip()
    df["UNIDAD"] = df["UNIDAD"].astype(str).str.upper().str.strip()
    df["PRODUCTO"] = df["PRODUCTO"].astype(str).str.strip()

    df = df.dropna(subset=["FECHA", "CANTIDAD"])
    df = df.sort_values("FECHA")

    return df

df = cargar_ventas()

if df.empty:
    st.warning("No existen datos en la tabla ventas.")
    st.stop()

df = df[df["SEDE"] == st.session_state.sede].copy()

if df.empty:
    st.warning("No existen ventas registradas para esta sede.")
    st.stop()

# ============================================================
# SIDEBAR CORPORATIVO
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">Market Donna</div>
    <div class="sidebar-subtitle">Sistema ejecutivo de pedidos</div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"**Usuario:** {st.session_state.usuario}")
    st.markdown(f"**Sede:** {st.session_state.sede}")

    st.divider()

    st.caption("Estado de datos")
    st.markdown(f"Desde: **{df['FECHA'].min().date()}**")
    st.markdown(f"Hasta: **{df['FECHA'].max().date()}**")
    st.markdown(f"Registros: **{len(df):,}**")

    st.divider()

    if st.button("Cerrar sesión", use_container_width=True):
        st.session_state.login = False
        st.session_state.usuario = ""
        st.session_state.sede = ""
        st.rerun()

# ============================================================
# HERO
# ============================================================

st.markdown(f"""
<div class="hero">
    <div class="hero-badge">● Modelo predictivo activo</div>
    <div class="hero-title">Dashboard de Predicción de Pedidos</div>
    <div class="hero-subtitle">
        Herramienta empresarial para estimar demanda, priorizar productos y generar
        hojas de pedido basadas en comportamiento histórico de ventas.
    </div>
    <br>
    <div class="user-chip">👤 {st.session_state.usuario} &nbsp;&nbsp; | &nbsp;&nbsp; 🏪 Sede {st.session_state.sede}</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INDICADORES
# ============================================================

total_kg = round(df["CANTIDAD"].sum(), 2)
total_productos = df["PRODUCTO"].nunique()
total_familias = df["FAMILIA"].nunique()
ultima_fecha = df["FECHA"].max().date()

c1, c2, c3, c4 = st.columns(4)

metricas = [
    ("📦", "Cantidad histórica", f"{total_kg:,.2f}"),
    ("🧾", "Productos registrados", f"{total_productos:,}"),
    ("🏷️", "Familias activas", f"{total_familias:,}"),
    ("📅", "Última venta", f"{ultima_fecha}")
]

for col, (icono, titulo, valor) in zip([c1, c2, c3, c4], metricas):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icono}</div>
            <div class="metric-label">{titulo}</div>
            <div class="metric-value">{valor}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ============================================================
# FILTROS
# ============================================================

st.markdown("""
<div class="soft-card">
    <div class="section-title">Centro de control</div>
    <div class="section-subtitle">
        Selecciona la familia y el horizonte para generar una hoja de pedido predictiva.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2.2, 1.5, 1.3])

with col1:
    familia = st.selectbox(
        "Familia de producto",
        sorted(df["FAMILIA"].dropna().unique())
    )

with col2:
    horizonte = st.selectbox(
        "Horizonte de predicción",
        [1, 7, 30],
        format_func=lambda x: "Mañana" if x == 1 else f"Próximos {x} días"
    )

df_familia = df[df["FAMILIA"] == familia].copy()

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='info-pill'>📌 {df_familia['PRODUCTO'].nunique()} productos detectados</span>",
        unsafe_allow_html=True
    )

# ============================================================
# FUNCIONES DE MACHINE LEARNING
# ============================================================

def preparar_producto(df_producto):

    datos = df_producto.copy()
    datos = datos.sort_values("FECHA")

    datos["year"] = datos["FECHA"].dt.year
    datos["month"] = datos["FECHA"].dt.month
    datos["day"] = datos["FECHA"].dt.day
    datos["dayofweek"] = datos["FECHA"].dt.dayofweek

    datos["lag1"] = datos["CANTIDAD"].shift(1)
    datos["lag7"] = datos["CANTIDAD"].shift(7)
    datos["lag14"] = datos["CANTIDAD"].shift(14)
    datos["media7"] = datos["CANTIDAD"].rolling(7).mean()
    datos["media14"] = datos["CANTIDAD"].rolling(14).mean()

    datos = datos.dropna()

    return datos


def entrenar_modelo(df_producto):

    datos = preparar_producto(df_producto)

    if len(datos) < 25:
        return None

    variables = [
        "year",
        "month",
        "day",
        "dayofweek",
        "lag1",
        "lag7",
        "lag14",
        "media7",
        "media14"
    ]

    X = datos[variables]
    y = datos["CANTIDAD"]

    modelo = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=10
    )

    modelo.fit(X, y)

    return modelo


def predecir_horizonte(modelo, df_producto, dias):

    historico = df_producto.copy()
    historico = historico.sort_values("FECHA")

    predicciones = []

    for _ in range(dias):

        fecha_predicha = historico["FECHA"].max() + pd.Timedelta(days=1)

        fila = pd.DataFrame({
            "year": [fecha_predicha.year],
            "month": [fecha_predicha.month],
            "day": [fecha_predicha.day],
            "dayofweek": [fecha_predicha.dayofweek],
            "lag1": [historico["CANTIDAD"].iloc[-1]],
            "lag7": [historico["CANTIDAD"].iloc[-7]],
            "lag14": [historico["CANTIDAD"].iloc[-14]],
            "media7": [historico["CANTIDAD"].tail(7).mean()],
            "media14": [historico["CANTIDAD"].tail(14).mean()]
        })

        pred = modelo.predict(fila)[0]
        pred = max(pred, 0)

        predicciones.append({
            "fecha_predicha": fecha_predicha,
            "prediccion": round(pred, 2)
        })

        nueva_fila = historico.iloc[-1].copy()
        nueva_fila["FECHA"] = fecha_predicha
        nueva_fila["CANTIDAD"] = pred

        historico = pd.concat(
            [historico, pd.DataFrame([nueva_fila])],
            ignore_index=True
        )

    return predicciones


def clasificar_recomendacion(prediccion, promedio):

    if prediccion > promedio * 1.20:
        return "Comprar más"
    elif prediccion < promedio * 0.80:
        return "Comprar menos"
    else:
        return "Compra normal"


def pintar_recomendacion(valor):
    if valor == "Comprar más":
        return "background-color: #DCFCE7; color: #166534; font-weight: 800;"
    if valor == "Comprar menos":
        return "background-color: #FEE2E2; color: #991B1B; font-weight: 800;"
    return "background-color: #DBEAFE; color: #1E40AF; font-weight: 800;"

# def generar_pdf(pedido, usuario, sede, familia, horizonte):

#     buffer = BytesIO()

#     doc = SimpleDocTemplate(
#         buffer,
#         pagesize=(21*cm,29.7*cm),
#         rightMargin=1.2*cm,
#         leftMargin=1.2*cm,
#         topMargin=1.5*cm,
#         bottomMargin=1.5*cm
#     )

#     estilos = getSampleStyleSheet()

#     titulo = estilos["Heading1"]
#     titulo.alignment = TA_CENTER

#     subtitulo = estilos["Heading2"]

#     normal = estilos["BodyText"]

#     elementos = []

#     elementos.append(Paragraph("<b>MARKET DONNA</b>", titulo))
#     elementos.append(Paragraph("Reporte Ejecutivo de Predicción de Pedidos", subtitulo))
#     elementos.append(Spacer(1,15))

#     elementos.append(Paragraph(
#         f"<b>Fecha de generación:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}",
#         normal))

#     elementos.append(Paragraph(
#         f"<b>Usuario:</b> {usuario}",
#         normal))

#     elementos.append(Paragraph(
#         f"<b>Sede:</b> {sede}",
#         normal))

#     elementos.append(Paragraph(
#         f"<b>Familia:</b> {familia}",
#         normal))

#     elementos.append(Paragraph(
#         f"<b>Horizonte:</b> {horizonte} día(s)",
#         normal))

#     elementos.append(Spacer(1,20))

#     elementos.append(Paragraph("<b>Resumen Ejecutivo</b>", subtitulo))

#     resumen = [
#         ["Indicador","Valor"],
#         ["Productos Analizados", str(len(pedido))],
#         ["Predicción Total", f"{pedido['PREDICCION_TOTAL'].sum():,.2f}"],
#         ["Pedido Sugerido", f"{pedido['PEDIDO_SUGERIDO'].sum():,.0f}"]
#     ]

#     tabla = Table(resumen, colWidths=[8*cm,6*cm])

#     tabla.setStyle(TableStyle([
#         ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1D4ED8")),
#         ('TEXTCOLOR',(0,0),(-1,0),colors.white),
#         ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
#         ('ALIGN',(0,0),(-1,-1),'CENTER'),
#         ('GRID',(0,0),(-1,-1),1,colors.grey),
#         ('BACKGROUND',(0,1),(-1,-1),colors.whitesmoke),
#         ('BOTTOMPADDING',(0,0),(-1,0),10),
#     ]))

#     elementos.append(tabla)

#     elementos.append(Spacer(1,20))

#     elementos.append(Paragraph("<b>Hoja de Pedido Recomendada</b>", subtitulo))

#     datos = [[
#         "Producto",
#         "Unidad",
#         "Predicción",
#         "Pedido",
#         "Recomendación"
#     ]]

#     for _, fila in pedido.iterrows():

#         datos.append([
#             fila["DESCRIPCIO"],
#             fila["UNIDAD"],
#             f"{fila['PREDICCION_TOTAL']:.2f}",
#             str(fila["PEDIDO_SUGERIDO"]),
#             fila["RECOMENDACION"]
#         ])

#     tabla2 = Table(datos)

#     tabla2.setStyle(TableStyle([
#         ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#10B981")),
#         ('TEXTCOLOR',(0,0),(-1,0),colors.white),
#         ('GRID',(0,0),(-1,-1),0.5,colors.grey),
#         ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
#         ('BACKGROUND',(0,1),(-1,-1),colors.beige),
#         ('FONTSIZE',(0,0),(-1,-1),8),
#         ('BOTTOMPADDING',(0,0),(-1,0),8)
#     ]))

#     elementos.append(tabla2)

#     elementos.append(Spacer(1,20))

#     elementos.append(Paragraph("<b>Observaciones</b>", subtitulo))

#     elementos.append(Paragraph(
#         """
#         • Este reporte fue generado automáticamente por el sistema
#         inteligente de predicción de pedidos de Market Donna.

#         • Las cantidades sugeridas fueron calculadas utilizando un modelo
#         Random Forest entrenado con el historial de ventas.

#         • Se recomienda revisar diariamente este reporte antes de realizar
#         las compras para abastecimiento.
#         """,
#         normal
#     ))

#     doc.build(elementos)

#     pdf = buffer.getvalue()
#     buffer.close()

#     return pdf

def generar_pdf(pedido, usuario, sede, familia, horizonte):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=0.8 * cm,
        leftMargin=0.8 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm
    )

    estilos = getSampleStyleSheet()

    titulo = ParagraphStyle(
        "TituloMarketDonna",
        parent=estilos["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        alignment=TA_CENTER,
        spaceAfter=4
    )

    subtitulo = ParagraphStyle(
        "SubtituloMarketDonna",
        parent=estilos["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=12
    )

    seccion = ParagraphStyle(
        "SeccionMarketDonna",
        parent=estilos["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=8
    )

    celda = ParagraphStyle(
        "CeldaMarketDonna",
        parent=estilos["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        textColor=colors.HexColor("#111827"),
        alignment=TA_LEFT
    )

    celda_bold = ParagraphStyle(
        "CeldaBoldMarketDonna",
        parent=celda,
        fontName="Helvetica-Bold"
    )

    elementos = []

    elementos.append(Paragraph("MARKET DONNA", titulo))
    elementos.append(Paragraph("Reporte Ejecutivo de Predicción de Pedidos", subtitulo))

    # =========================
    # INFORMACIÓN GENERAL
    # =========================

    info = [
        [
            Paragraph("<b>Fecha de generación</b>", celda_bold),
            datetime.now().strftime("%d/%m/%Y %H:%M"),
            Paragraph("<b>Usuario</b>", celda_bold),
            escape(str(usuario))
        ],
        [
            Paragraph("<b>Sede</b>", celda_bold),
            escape(str(sede)),
            Paragraph("<b>Familia</b>", celda_bold),
            escape(str(familia))
        ],
        [
            Paragraph("<b>Horizonte</b>", celda_bold),
            f"{horizonte} día(s)",
            Paragraph("<b>Productos analizados</b>", celda_bold),
            str(len(pedido))
        ]
    ]

    tabla_info = Table(
        info,
        colWidths=[4.2 * cm, 6.0 * cm, 4.2 * cm, 6.0 * cm]
    )

    tabla_info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elementos.append(tabla_info)
    elementos.append(Spacer(1, 10))

    # =========================
    # RESUMEN EJECUTIVO
    # =========================

    elementos.append(Paragraph("Resumen Ejecutivo", seccion))

    resumen = [
        [
            "Productos analizados",
            str(len(pedido)),
            "Predicción total",
            f"{pedido['PREDICCION_TOTAL'].sum():,.2f}",
            "Pedido sugerido",
            f"{pedido['PEDIDO_SUGERIDO'].sum():,.0f}"
        ]
    ]

    tabla_resumen = Table(
        resumen,
        colWidths=[3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm]
    )

    tabla_resumen.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1E3A8A")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    elementos.append(tabla_resumen)
    elementos.append(Spacer(1, 12))

    # =========================
    # TABLA PRINCIPAL
    # =========================

    elementos.append(Paragraph("Hoja de Pedido Recomendada", seccion))

    datos = [[
        "PRODUCTO",
        "DESCRIPCIO",
        "UNIDAD",
        "FAMILIA",
        "HORIZONTE_DIAS",
        "PREDICCION_TOTAL",
        "PEDIDO_SUGERIDO",
        "RECOMENDACION"
    ]]

    estilos_recomendacion = []

    for _, fila in pedido.iterrows():

        datos.append([
            Paragraph(escape(str(fila["PRODUCTO"])), celda),
            Paragraph(escape(str(fila["DESCRIPCIO"])), celda),
            Paragraph(escape(str(fila["UNIDAD"])), celda),
            Paragraph(escape(str(fila["FAMILIA"])), celda),
            str(fila["HORIZONTE_DIAS"]),
            f"{fila['PREDICCION_TOTAL']:,.2f}",
            f"{fila['PEDIDO_SUGERIDO']:,.0f}",
            Paragraph(escape(str(fila["RECOMENDACION"])), celda_bold)
        ])

        fila_pdf = len(datos) - 1

        if fila["RECOMENDACION"] == "Comprar más":
            estilos_recomendacion.extend([
                ("BACKGROUND", (7, fila_pdf), (7, fila_pdf), colors.HexColor("#DCFCE7")),
                ("TEXTCOLOR", (7, fila_pdf), (7, fila_pdf), colors.HexColor("#166534")),
                ("FONTNAME", (7, fila_pdf), (7, fila_pdf), "Helvetica-Bold"),
            ])

        elif fila["RECOMENDACION"] == "Comprar menos":
            estilos_recomendacion.extend([
                ("BACKGROUND", (7, fila_pdf), (7, fila_pdf), colors.HexColor("#FEE2E2")),
                ("TEXTCOLOR", (7, fila_pdf), (7, fila_pdf), colors.HexColor("#991B1B")),
                ("FONTNAME", (7, fila_pdf), (7, fila_pdf), "Helvetica-Bold"),
            ])

        else:
            estilos_recomendacion.extend([
                ("BACKGROUND", (7, fila_pdf), (7, fila_pdf), colors.HexColor("#DBEAFE")),
                ("TEXTCOLOR", (7, fila_pdf), (7, fila_pdf), colors.HexColor("#1E40AF")),
                ("FONTNAME", (7, fila_pdf), (7, fila_pdf), "Helvetica-Bold"),
            ])

    tabla_pedido = Table(
        datos,
        colWidths=[
            1.8 * cm,   # PRODUCTO
            6.7 * cm,   # DESCRIPCIO
            1.8 * cm,   # UNIDAD
            1.8 * cm,   # FAMILIA
            2.3 * cm,   # HORIZONTE
            2.7 * cm,   # PREDICCION
            2.6 * cm,   # PEDIDO
            3.0 * cm    # RECOMENDACION
        ],
        repeatRows=1
    )

    estilo_base = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#475569")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),

        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5E7EB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F9FAFB")
        ]),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 1), (-1, -1), 7),

        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("ALIGN", (4, 1), (6, -1), "RIGHT"),
        ("ALIGN", (7, 1), (7, -1), "CENTER"),

        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]

    tabla_pedido.setStyle(TableStyle(estilo_base + estilos_recomendacion))

    elementos.append(tabla_pedido)
    elementos.append(Spacer(1, 12))

    # =========================
    # OBSERVACIONES
    # =========================

    elementos.append(Paragraph("Observaciones", seccion))

    observaciones = """
    Este reporte fue generado automáticamente por el sistema inteligente de predicción de pedidos de Market Donna.
    La predicción total y el pedido sugerido fueron calculados mediante un modelo Random Forest entrenado con el historial de ventas.
    """

    elementos.append(Paragraph(observaciones, celda))

    doc.build(elementos)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf

# ============================================================
# GENERAR PREDICCIÓN
# ============================================================

st.markdown("""
<div class="soft-card">
    <div class="section-title">Generación de hoja de pedido</div>
    <div class="section-subtitle">
        El sistema entrenará un modelo por producto y generará una propuesta de compra según la demanda estimada.
    </div>
</div>
""", unsafe_allow_html=True)

generar = st.button("🚀 Generar predicción empresarial", use_container_width=True)

if generar:

    resultados = []
    registros_supabase = []

    productos = df_familia["PRODUCTO"].dropna().unique()
    barra = st.progress(0, text="Entrenando modelos y calculando predicciones...")

    for i, producto in enumerate(productos):

        datos_producto = df_familia[df_familia["PRODUCTO"] == producto].copy()

        if len(datos_producto) < 30:
            barra.progress((i + 1) / len(productos), text=f"Procesando productos... {i + 1}/{len(productos)}")
            continue

        modelo = entrenar_modelo(datos_producto)

        if modelo is None:
            barra.progress((i + 1) / len(productos), text=f"Procesando productos... {i + 1}/{len(productos)}")
            continue

        predicciones = predecir_horizonte(modelo, datos_producto, horizonte)

        total_predicho = sum([p["prediccion"] for p in predicciones])
        promedio_historico = datos_producto["CANTIDAD"].tail(14).mean()

        recomendacion = clasificar_recomendacion(total_predicho / horizonte, promedio_historico)
        pedido_sugerido = round(total_predicho, 0)

        descripcion = datos_producto["DESCRIPCIO"].iloc[-1]
        unidad = datos_producto["UNIDAD"].iloc[-1]
    

        resultados.append({
            "PRODUCTO": producto,
            "DESCRIPCIO": descripcion,
            "UNIDAD": unidad,
            "FAMILIA": familia,
            "HORIZONTE_DIAS": horizonte,
            "PREDICCION_TOTAL": round(total_predicho, 2),
            "PEDIDO_SUGERIDO": int(pedido_sugerido),
            "RECOMENDACION": recomendacion
        })

        for pred in predicciones:
            registros_supabase.append({
                "fecha_generacion": datetime.now().date().isoformat(),
                "fecha_predicha": pred["fecha_predicha"].date().isoformat(),
                "producto": str(producto),
                "descripcio": descripcion,
                "unidad": unidad,
                "familia": familia,
                "sede": st.session_state.sede,
                "prediccion": pred["prediccion"],
                "pedido_sugerido": int(round(pred["prediccion"], 0)),
                "horizonte": horizonte
            })

        barra.progress((i + 1) / len(productos), text=f"Procesando productos... {i + 1}/{len(productos)}")

    pedido = pd.DataFrame(resultados)

    if pedido.empty:
        st.warning("No hay suficiente historial para generar predicciones.")
        st.stop()

    pedido = pedido.sort_values("PREDICCION_TOTAL", ascending=False)

    columnas_orden = [
    "PRODUCTO",
    "DESCRIPCIO",
    "UNIDAD",
    "FAMILIA",
    "HORIZONTE_DIAS",
    "PREDICCION_TOTAL",
    "PEDIDO_SUGERIDO",
    "RECOMENDACION"
    ]
    
    pedido = pedido[columnas_orden]
    
    st.success("Predicción generada correctamente.")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Productos con predicción", len(pedido))

    with r2:
        st.metric("Pedido sugerido total", f"{pedido['PEDIDO_SUGERIDO'].sum():,.0f}")

    with r3:
        st.metric("Predicción total", f"{pedido['PREDICCION_TOTAL'].sum():,.2f}")

    st.write("")

    st.markdown("""
    <div class="section-title">Hoja de pedido recomendada</div>
    <div class="section-subtitle">
        Tabla ordenada de mayor a menor demanda esperada.
    </div>
    """, unsafe_allow_html=True)

    try:
        pedido_vista = pedido.style.map(
            pintar_recomendacion,
            subset=["RECOMENDACION"]
        ).format({
            
            "PREDICCION_TOTAL": "{:,.2f}",
            "PEDIDO_SUGERIDO": "{:,.0f}"
        })
    
        st.dataframe(pedido_vista, use_container_width=True, height=430)

    except Exception:
        st.dataframe(pedido, use_container_width=True, height=430)

    archivo = pedido.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Descargar hoja de pedido CSV",
        data=archivo,
        file_name=f"pedido_{familia}_{horizonte}_dias.csv",
        mime="text/csv",
        use_container_width=True
    )

    pdf = generar_pdf(
        pedido,
        st.session_state.usuario,
        st.session_state.sede,
        familia,
        horizonte
    )

    st.download_button(
        label="📄 Descargar Reporte Ejecutivo PDF",
        data=pdf,
        file_name=f"Reporte_MarketDonna_{familia}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.write("")

    st.markdown("""
    <div class="section-title">Productos prioritarios</div>
    <div class="section-subtitle">
        Top 10 productos con mayor predicción total.
    </div>
    """, unsafe_allow_html=True)

    top = pedido.head(10).set_index("DESCRIPCIO")["PREDICCION_TOTAL"]
    st.bar_chart(top)
# ============================================================
# ANÁLISIS HISTÓRICO
# ============================================================

st.write("")
st.markdown("""
<div class="soft-card">
    <div class="section-title">Análisis histórico</div>
    <div class="section-subtitle">
        Consulta el comportamiento de ventas, productos con mayor rotación y registros históricos.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📈 Ventas por fecha",
    "🏆 Top productos",
    "🗂️ Datos históricos"
])

with tab1:
    ventas_fecha = (
        df_familia
        .groupby("FECHA")["CANTIDAD"]
        .sum()
        .reset_index()
    )

    st.line_chart(
        ventas_fecha,
        x="FECHA",
        y="CANTIDAD",
        use_container_width=True
    )

with tab2:
    top_productos = (
        df_familia
        .groupby("DESCRIPCIO")["CANTIDAD"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(top_productos, use_container_width=True)

with tab3:
    st.dataframe(
        df_familia.sort_values("FECHA", ascending=False),
        use_container_width=True,
        height=520
    )
