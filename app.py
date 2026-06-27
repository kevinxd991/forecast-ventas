# app.py

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Market Donna",
    page_icon="🛒",
    layout="wide"
)

# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F6F9;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #1F2937;
}

.subtitle {
    font-size: 18px;
    color: #6B7280;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.metric-title {
    color: #6B7280;
    font-size: 15px;
}

.metric-value {
    color: #111827;
    font-size: 28px;
    font-weight: 800;
}

.stButton > button {
    background-color: #0F62FE;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 48px;
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
# LOGIN
# ============================================================

if not st.session_state.login:

    st.markdown("<p class='main-title'>🛒 Market Donna</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Sistema inteligente de predicción de pedidos</p>", unsafe_allow_html=True)

    usuario = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Ingresar", use_container_width=True):

        respuesta = (
            supabase
            .table("usuarios")
            .select("*")
            .eq("usuario", usuario)
            .execute()
        )

        if len(respuesta.data) == 0:
            st.error("Usuario no encontrado")
            st.stop()

        datos = respuesta.data[0]

        if datos["password"] != password:
            st.error("Contraseña incorrecta")
            st.stop()

        st.session_state.login = True
        st.session_state.usuario = datos["usuario"]
        st.session_state.sede = datos["sede"]

        st.rerun()

    st.stop()

# ============================================================
# CARGAR DATOS
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

    df = df.dropna(subset=["FECHA", "CANTIDAD"])
    df = df.sort_values("FECHA")

    return df

# @st.cache_data(ttl=120)
# def cargar_ventas():

#     respuesta = (
#         supabase
#         .table("ventas")
#         .select("*")
#         .execute()
#     )

#     df = pd.DataFrame(respuesta.data)

#     if df.empty:
#         return df

#     df.columns = df.columns.str.upper()

#     df["FECHA"] = pd.to_datetime(df["FECHA"])
#     df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce")
#     df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")

#     df = df.dropna(subset=["FECHA", "CANTIDAD"])
#     df = df.sort_values("FECHA")

#     return df


df = cargar_ventas()
st.info(f"Datos cargados desde {df['FECHA'].min().date()} hasta {df['FECHA'].max().date()}")
if df.empty:
    st.warning("No existen datos en la tabla ventas.")
    st.stop()

# Filtrar por sede del usuario
df = df[df["SEDE"] == st.session_state.sede].copy()

if df.empty:
    st.warning("No existen ventas registradas para esta sede.")
    st.stop()

# ============================================================
# CABECERA
# ============================================================

st.markdown("<p class='main-title'>📊 Dashboard de Predicción de Pedidos</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistema de apoyo para la toma de decisiones de compra</p>", unsafe_allow_html=True)

col_user, col_sede, col_logout = st.columns([2, 2, 1])

with col_user:
    st.success(f"👤 Usuario: {st.session_state.usuario}")

with col_sede:
    st.success(f"🏪 Sede: {st.session_state.sede}")

with col_logout:
    if st.button("Salir"):
        st.session_state.login = False
        st.rerun()

st.divider()

# ============================================================
# INDICADORES
# ============================================================

total_kg = round(df["CANTIDAD"].sum(), 2)
total_productos = df["PRODUCTO"].nunique()
total_familias = df["FAMILIA"].nunique()
ultima_fecha = df["FECHA"].max().date()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Cantidad histórica vendida</div>
        <div class="metric-value">{total_kg}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Productos registrados</div>
        <div class="metric-value">{total_productos}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Familias</div>
        <div class="metric-value">{total_familias}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Última fecha de venta</div>
        <div class="metric-value">{ultima_fecha}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================
# FILTROS
# ============================================================

st.subheader("🔎 Filtros de análisis")

col1, col2 = st.columns(2)

with col1:
    familia = st.selectbox(
        "Seleccione familia",
        sorted(df["FAMILIA"].dropna().unique())
    )

with col2:
    horizonte = st.selectbox(
        "Horizonte de predicción",
        [1, 7, 30],
        format_func=lambda x: "Mañana" if x == 1 else f"Próximos {x} días"
    )

df_familia = df[df["FAMILIA"] == familia].copy()

st.info(f"Productos encontrados en {familia}: {df_familia['PRODUCTO'].nunique()}")

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

    for i in range(dias):

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


# ============================================================
# GENERAR PREDICCIÓN
# ============================================================

st.subheader("🚀 Generación de hoja de pedido")

if st.button("Generar predicción", use_container_width=True):

    resultados = []
    registros_supabase = []

    # productos = df_familia["PRODUCTO"].unique()
    productos = df_familia["PRODUCTO"].dropna().unique()
    barra = st.progress(0)

    for i, producto in enumerate(productos):

        datos_producto = df_familia[df_familia["PRODUCTO"] == producto].copy()

        if len(datos_producto) < 30:
            continue

        modelo = entrenar_modelo(datos_producto)

        if modelo is None:
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

        barra.progress((i + 1) / len(productos))

    pedido = pd.DataFrame(resultados)

    if pedido.empty:
        st.warning("No hay suficiente historial para generar predicciones.")
        st.stop()

    pedido = pedido.sort_values("PREDICCION_TOTAL", ascending=False)

    # Guardar en Supabase
    # try:
    #     supabase.table("predicciones").insert(registros_supabase).execute()
    #     st.success("Predicción generada y guardada correctamente en Supabase.")
    # except Exception as e:
    #     st.warning("La predicción se generó, pero no se pudo guardar en Supabase.")
    #     st.write(e)
st.success("Predicción generada correctamente.")
    # Mostrar tabla
    st.dataframe(pedido, use_container_width=True)

    # Descargar Excel
    archivo = pedido.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="📥 Descargar hoja de pedido CSV",
        data=archivo,
        file_name=f"pedido_{familia}_{horizonte}_dias.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.divider()

    # Gráfico top productos
    st.subheader("📌 Productos con mayor pedido sugerido")

    top = pedido.head(10).set_index("DESCRIPCIO")["PREDICCION_TOTAL"]

    st.bar_chart(top)

# ============================================================
# ANÁLISIS HISTÓRICO
# ============================================================

st.divider()
st.subheader("📈 Análisis histórico")

tab1, tab2, tab3 = st.tabs([
    "Ventas por fecha",
    "Top productos",
    "Datos históricos"
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
        y="CANTIDAD"
    )

with tab2:
    top_productos = (
        df_familia
        .groupby("DESCRIPCIO")["CANTIDAD"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    st.bar_chart(top_productos)

with tab3:
    st.dataframe(
        df_familia.sort_values("FECHA", ascending=False),
        use_container_width=True
    )

# # ============================================================
# # MARKET DONNA
# # Sistema Inteligente de Predicción de Pedidos
# # ============================================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# from supabase import create_client
# from datetime import datetime
# import joblib
# import os

# # Machine Learning
# from sklearn.ensemble import RandomForestRegressor

# # ============================================================
# # CONFIGURACIÓN
# # ============================================================

# st.set_page_config(
#     page_title="Market Donna",
#     page_icon="🛒",
#     layout="wide"
# )

# # ============================================================
# # ESTILOS
# # ============================================================

# st.markdown("""
# <style>

# .stApp{

# background:#F4F6F9;

# }

# .titulo{

# font-size:40px;
# font-weight:bold;
# color:#1F2937;

# }

# .subtitulo{

# font-size:18px;
# color:gray;

# }

# div.stButton > button{

# background:#0F62FE;
# color:white;
# font-weight:bold;
# border-radius:10px;
# height:50px;

# }

# </style>
# """,unsafe_allow_html=True)

# # ============================================================
# # CONEXIÓN SUPABASE
# # ============================================================

# SUPABASE_URL=st.secrets["supabase"]["url"]

# SUPABASE_KEY=st.secrets["supabase"]["key"]

# supabase=create_client(
# SUPABASE_URL,
# SUPABASE_KEY
# )

# # ============================================================
# # VARIABLES DE SESIÓN
# # ============================================================

# if "login" not in st.session_state:

#     st.session_state.login=False

# if "usuario" not in st.session_state:

#     st.session_state.usuario=""

# if "sede" not in st.session_state:

#     st.session_state.sede=""

# # ============================================================
# # LOGIN
# # ============================================================

# if st.session_state.login==False:

#     st.markdown("<p class='titulo'>Market Donna</p>",unsafe_allow_html=True)

#     st.markdown("<p class='subtitulo'>Sistema Inteligente de Predicción de Pedidos</p>",unsafe_allow_html=True)

#     usuario=st.text_input("Usuario")

#     password=st.text_input(
#         "Contraseña",
#         type="password"
#     )

#     if st.button("Ingresar"):

#         respuesta=(supabase
#                    .table("usuarios")
#                    .select("*")
#                    .eq("usuario",usuario)
#                    .execute())

#         if len(respuesta.data)==0:

#             st.error("Usuario no encontrado")

#             st.stop()

#         datos=respuesta.data[0]

#         if datos["password"]!=password:

#             st.error("Contraseña incorrecta")

#             st.stop()

#         st.session_state.login=True

#         st.session_state.usuario=datos["usuario"]

#         st.session_state.sede=datos["sede"]

#         st.rerun()

#     st.stop()

# # ============================================================
# # DASHBOARD
# # ============================================================

# st.markdown("<p class='titulo'>Sistema Inteligente de Pedidos</p>", unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     st.success(f"👤 Usuario : {st.session_state.usuario}")

# with col2:
#     st.success(f"🏪 Sede : {st.session_state.sede}")

# st.divider()

# # ============================================================
# # CARGAR VENTAS DESDE SUPABASE
# # ============================================================

# @st.cache_data(ttl=60)
# def cargar_ventas():

#     respuesta = (
#         supabase
#         .table("ventas")
#         .select("*")
#         .limit(10)
#         .execute()
#     )

#     df = pd.DataFrame(respuesta.data)

#     return df

# df = cargar_ventas()

# st.write(df.head())
# st.write(df.columns.tolist())

# if df.empty:

#     st.warning("No existen ventas registradas para esta sede.")

#     st.stop()

# # ============================================================
# # LIMPIEZA
# # ============================================================

# df.columns=df.columns.str.upper()

# df["FECHA"]=pd.to_datetime(df["FECHA"])

# df["CANTIDAD"]=pd.to_numeric(df["CANTIDAD"])

# df=df.sort_values("FECHA")

# # ============================================================
# # FAMILIAS
# # ============================================================

# familias=sorted(df["FAMILIA"].dropna().unique())

# familia=st.selectbox(

#     "Seleccione una familia",

#     familias

# )

# df_familia=df[df["FAMILIA"]==familia].copy()

# st.write("")

# st.info(f"Productos encontrados : {df_familia['PRODUCTO'].nunique()}")

# st.write("")

# # ============================================================
# # BOTÓN PRINCIPAL
# # ============================================================

# generar=st.button(

#     "🚀 GENERAR HOJA DE PEDIDO",

#     use_container_width=True

# )
# # ============================================================
# # FUNCION PARA PREPARAR LOS DATOS DE UN PRODUCTO
# # ============================================================

# def preparar_producto(df_producto):

#     df=df_producto.copy()

#     df=df.sort_values("FECHA")

#     df["year"]=df["FECHA"].dt.year
#     df["month"]=df["FECHA"].dt.month
#     df["day"]=df["FECHA"].dt.day
#     df["dayofweek"]=df["FECHA"].dt.dayofweek

#     df["lag1"]=df["CANTIDAD"].shift(1)
#     df["lag7"]=df["CANTIDAD"].shift(7)
#     df["lag14"]=df["CANTIDAD"].shift(14)

#     df["media7"]=df["CANTIDAD"].rolling(7).mean()

#     df=df.dropna()

#     return df


# # ============================================================
# # ENTRENAR UN PRODUCTO
# # ============================================================

# def entrenar_producto(df_producto):

#     df=preparar_producto(df_producto)

#     if len(df)<20:

#         return None

#     variables=[
#         "year",
#         "month",
#         "day",
#         "dayofweek",
#         "lag1",
#         "lag7",
#         "lag14",
#         "media7"
#     ]

#     X=df[variables]

#     y=df["CANTIDAD"]

#     modelo=RandomForestRegressor(

#         n_estimators=200,

#         random_state=42

#     )

#     modelo.fit(X,y)

#     return modelo
# # ============================================================
# # PREDECIR EL DIA SIGUIENTE
# # ============================================================

# def predecir_manana(modelo,df_producto):

#     df=df_producto.copy()

#     df=df.sort_values("FECHA")

#     ultima=df.iloc[-1]

#     mañana=ultima["FECHA"]+pd.Timedelta(days=1)

#     fila=pd.DataFrame({

#         "year":[mañana.year],

#         "month":[mañana.month],

#         "day":[mañana.day],

#         "dayofweek":[mañana.dayofweek],

#         "lag1":[df["CANTIDAD"].iloc[-1]],

#         "lag7":[df["CANTIDAD"].iloc[-7]],

#         "lag14":[df["CANTIDAD"].iloc[-14]],

#         "media7":[df["CANTIDAD"].tail(7).mean()]

#     })

#     pred=modelo.predict(fila)[0]

#     if pred<0:

#         pred=0

#     return round(pred,2)
# # ============================================================
# # GENERAR HOJA
# # ============================================================

# if generar:

#     hoja=[]

#     productos=df_familia["PRODUCTO"].unique()

#     barra=st.progress(0)

#     total=len(productos)

#     for i,producto in enumerate(productos):

#         datos=df_familia[

#             df_familia["PRODUCTO"]==producto

#         ].copy()

#         if len(datos)<25:

#             continue

#         modelo=entrenar_producto(datos)

#         if modelo is None:

#             continue

#         pred=predecir_manana(

#             modelo,

#             datos

#         )

#         hoja.append({

#             "PRODUCTO":producto,

#             "DESCRIPCIO":datos["DESCRIPCIO"].iloc[0],

#             "UNIDAD":datos["UNIDAD"].iloc[0],

#             "PEDIDO":pred

#         })

#         barra.progress((i+1)/total)

#     pedido=pd.DataFrame(hoja)

#     pedido=pedido.sort_values(

#         "DESCRIPCIO"

#     )

#     st.success("Hoja generada correctamente")

#     st.dataframe(

#         pedido,

#         use_container_width=True

#     )
