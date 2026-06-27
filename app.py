# ============================================================
# MARKET DONNA
# Sistema Inteligente de Predicción de Pedidos
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client
from datetime import datetime
import joblib
import os

# Machine Learning
from sklearn.ensemble import RandomForestRegressor

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

.stApp{

background:#F4F6F9;

}

.titulo{

font-size:40px;
font-weight:bold;
color:#1F2937;

}

.subtitulo{

font-size:18px;
color:gray;

}

div.stButton > button{

background:#0F62FE;
color:white;
font-weight:bold;
border-radius:10px;
height:50px;

}

</style>
""",unsafe_allow_html=True)

# ============================================================
# CONEXIÓN SUPABASE
# ============================================================

SUPABASE_URL=st.secrets["supabase"]["url"]

SUPABASE_KEY=st.secrets["supabase"]["key"]

supabase=create_client(
SUPABASE_URL,
SUPABASE_KEY
)

# ============================================================
# VARIABLES DE SESIÓN
# ============================================================

if "login" not in st.session_state:

    st.session_state.login=False

if "usuario" not in st.session_state:

    st.session_state.usuario=""

if "sede" not in st.session_state:

    st.session_state.sede=""

# ============================================================
# LOGIN
# ============================================================

if st.session_state.login==False:

    st.markdown("<p class='titulo'>Market Donna</p>",unsafe_allow_html=True)

    st.markdown("<p class='subtitulo'>Sistema Inteligente de Predicción de Pedidos</p>",unsafe_allow_html=True)

    usuario=st.text_input("Usuario")

    password=st.text_input(
        "Contraseña",
        type="password"
    )

    if st.button("Ingresar"):

        respuesta=(supabase
                   .table("usuarios")
                   .select("*")
                   .eq("usuario",usuario)
                   .execute())

        if len(respuesta.data)==0:

            st.error("Usuario no encontrado")

            st.stop()

        datos=respuesta.data[0]

        if datos["password"]!=password:

            st.error("Contraseña incorrecta")

            st.stop()

        st.session_state.login=True

        st.session_state.usuario=datos["usuario"]

        st.session_state.sede=datos["sede"]

        st.rerun()

    st.stop()

# ============================================================
# DASHBOARD
# ============================================================

st.markdown("<p class='titulo'>Sistema Inteligente de Pedidos</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.success(f"👤 Usuario : {st.session_state.usuario}")

with col2:
    st.success(f"🏪 Sede : {st.session_state.sede}")

st.divider()

# ============================================================
# CARGAR VENTAS DESDE SUPABASE
# ============================================================

@st.cache_data(ttl=60)
def cargar_ventas():

    respuesta = (
        supabase
        .table("ventas")
        .select("*")
        .limit(10)
        .execute()
    )

    df = pd.DataFrame(respuesta.data)

    return df

df = cargar_ventas()

st.write(df.head())
st.write(df.columns.tolist())

if df.empty:

    st.warning("No existen ventas registradas para esta sede.")

    st.stop()

# ============================================================
# LIMPIEZA
# ============================================================

df.columns=df.columns.str.upper()

df["FECHA"]=pd.to_datetime(df["FECHA"])

df["CANTIDAD"]=pd.to_numeric(df["CANTIDAD"])

df=df.sort_values("FECHA")

# ============================================================
# FAMILIAS
# ============================================================

familias=sorted(df["FAMILIA"].dropna().unique())

familia=st.selectbox(

    "Seleccione una familia",

    familias

)

df_familia=df[df["FAMILIA"]==familia].copy()

st.write("")

st.info(f"Productos encontrados : {df_familia['PRODUCTO'].nunique()}")

st.write("")

# ============================================================
# BOTÓN PRINCIPAL
# ============================================================

generar=st.button(

    "🚀 GENERAR HOJA DE PEDIDO",

    use_container_width=True

)
# ============================================================
# FUNCION PARA PREPARAR LOS DATOS DE UN PRODUCTO
# ============================================================

def preparar_producto(df_producto):

    df=df_producto.copy()

    df=df.sort_values("FECHA")

    df["year"]=df["FECHA"].dt.year
    df["month"]=df["FECHA"].dt.month
    df["day"]=df["FECHA"].dt.day
    df["dayofweek"]=df["FECHA"].dt.dayofweek

    df["lag1"]=df["CANTIDAD"].shift(1)
    df["lag7"]=df["CANTIDAD"].shift(7)
    df["lag14"]=df["CANTIDAD"].shift(14)

    df["media7"]=df["CANTIDAD"].rolling(7).mean()

    df=df.dropna()

    return df


# ============================================================
# ENTRENAR UN PRODUCTO
# ============================================================

def entrenar_producto(df_producto):

    df=preparar_producto(df_producto)

    if len(df)<20:

        return None

    variables=[
        "year",
        "month",
        "day",
        "dayofweek",
        "lag1",
        "lag7",
        "lag14",
        "media7"
    ]

    X=df[variables]

    y=df["CANTIDAD"]

    modelo=RandomForestRegressor(

        n_estimators=200,

        random_state=42

    )

    modelo.fit(X,y)

    return modelo
# ============================================================
# PREDECIR EL DIA SIGUIENTE
# ============================================================

def predecir_manana(modelo,df_producto):

    df=df_producto.copy()

    df=df.sort_values("FECHA")

    ultima=df.iloc[-1]

    mañana=ultima["FECHA"]+pd.Timedelta(days=1)

    fila=pd.DataFrame({

        "year":[mañana.year],

        "month":[mañana.month],

        "day":[mañana.day],

        "dayofweek":[mañana.dayofweek],

        "lag1":[df["CANTIDAD"].iloc[-1]],

        "lag7":[df["CANTIDAD"].iloc[-7]],

        "lag14":[df["CANTIDAD"].iloc[-14]],

        "media7":[df["CANTIDAD"].tail(7).mean()]

    })

    pred=modelo.predict(fila)[0]

    if pred<0:

        pred=0

    return round(pred,2)
# ============================================================
# GENERAR HOJA
# ============================================================

if generar:

    hoja=[]

    productos=df_familia["PRODUCTO"].unique()

    barra=st.progress(0)

    total=len(productos)

    for i,producto in enumerate(productos):

        datos=df_familia[

            df_familia["PRODUCTO"]==producto

        ].copy()

        if len(datos)<25:

            continue

        modelo=entrenar_producto(datos)

        if modelo is None:

            continue

        pred=predecir_manana(

            modelo,

            datos

        )

        hoja.append({

            "PRODUCTO":producto,

            "DESCRIPCIO":datos["DESCRIPCIO"].iloc[0],

            "UNIDAD":datos["UNIDAD"].iloc[0],

            "PEDIDO":pred

        })

        barra.progress((i+1)/total)

    pedido=pd.DataFrame(hoja)

    pedido=pedido.sort_values(

        "DESCRIPCIO"

    )

    st.success("Hoja generada correctamente")

    st.dataframe(

        pedido,

        use_container_width=True

    )
