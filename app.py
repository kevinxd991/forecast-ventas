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
