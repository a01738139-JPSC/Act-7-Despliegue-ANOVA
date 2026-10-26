#Creamos el archivo de la APP en el interprete principal (Phyton)
#####################################################
#Importamos librerias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
######################################################
#Configuración de la página (título, ícono y ancho)
st.set_page_config(page_title="ANOVA CLV", page_icon="📊", layout="wide")
######################################################
#Creamos la función de carga de datos
@st.cache_data
def load_data():
   #Lectura del archivo excel
   df=pd.read_excel("Datos_Complementarios_CLV.xlsx", sheet_name="Histórico_Entradas_SAlidas")
   #Rellenamos nulos (por si los hubiera)
   df =df.bfill()
   df =df.ffill()
   #Transformaciones
   df['SKU'] = df['SKU'].astype(str)   #SKU tenía números, texto y una fecha mezclados
   df['ANO'] = df['ANO'].astype(int)
   df['MES'] = df['MES'].astype(int)
   df = df[df['ANO'] >= 2021]          #2019 solo tiene 96 registros y no hay 2020
   return df
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("📊 CLV - ANOVA")

#Widget 1: Selectbox con las páginas
View= st.sidebar.selectbox(label= "Tipo de Análisis", options= ["ANOVA", "ANOVA Factorial"])

#Variables del análisis
Variable_y= "SALIDAS"                    #Única variable numérica real
Lista_cat= ['MARCA', 'ANO', 'MES']       #Variables categóricas útiles

#Widget 2: Slider para quedarnos con las marcas más frecuentes
top_n= st.sidebar.slider("Número de marcas (las más vendidas)", min_value=3, max_value=15, value=5)
top_marcas= df['MARCA'].value_counts().head(top_n).index
df_f= df[df['MARCA'].isin(top_marcas)].copy()

#Widget 3: Checkbox para aplicar logaritmo a SALIDAS (datos muy sesgados)
if st.sidebar.checkbox("Aplicar log(SALIDAS)"):
    df_f[Variable_y]= np.log1p(df_f[Variable_y])

####################################################################################
# CONTENIDO DE LA VISTA 1: ANOVA de un factor
if View == "ANOVA":
    Variable_x= st.sidebar.selectbox(label="Variable independiente (x)", options= Lista_cat)

    st.title("ANOVA de un factor")
    st.caption(f"¿Cambia el promedio de **{Variable_y}** según **{Variable_x}**?")

    #Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)

    # Contenedor A: ANOVA y resultados
    with Contenedor_A:
        st.subheader("Resultados del ANOVA")
        grupos = [grupo[Variable_y] for nombre, grupo in df_f.groupby(Variable_x)]
        f_valor, p_valor = stats.f_oneway(*grupos)

        col1, col2 = st.columns(2)
        col1.metric("F-valor", f"{f_valor:.3f}")
        col2.metric("p-valor", f"{p_valor:.4f}")
        if p_valor < 0.05:
            st.success("✅ ¡Hay diferencias significativas entre los grupos!")
        else:
            st.warning("⚠️ No hay diferencias significativas entre los grupos.")

        #Tabla resumen por grupo
        st.subheader("Tabla resumen")
        resumen = df_f.groupby(Variable_x)[Variable_y].agg(['count', 'mean', 'std']).round(2)
        st.dataframe(resumen)

    # Contenedor B: Gráfico de barras
    with Contenedor_B:
        st.subheader("Media por grupo")
        resumen = df_f.groupby(Variable_x)[Variable_y].agg(['mean', 'sem']).reset_index()

        fig, ax = plt.subplots()
        x = resumen[Variable_x].astype(str)
        ax.bar(x, resumen['mean'], yerr=resumen['sem'], capsize=5, color='skyblue', edgecolor='black')
        ax.set_xlabel(Variable_x)
        ax.set_ylabel(f"Media de {Variable_y}")
        ax.set_title("Media por grupo con Error Estándar")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    #Fila 2: Boxplot
    st.subheader("Boxplot por grupo")
    fig_box = px.box(df_f, x=df_f[Variable_x].astype(str), y=Variable_y, color=df_f[Variable_x].astype(str),
                     labels={"x": Variable_x, "color": Variable_x})
    st.plotly_chart(fig_box)

####################################################################################
# CONTENIDO DE LA VISTA 2: ANOVA Factorial (dos factores)
if View == "ANOVA Factorial":
    Factor_1= st.sidebar.selectbox(label="Factor 1", options= Lista_cat, index=0)
    Factor_2= st.sidebar.selectbox(label="Factor 2", options= [c for c in Lista_cat if c != Factor_1])

    st.title("ANOVA Factorial (dos factores)")
    st.caption(f"Efecto de **{Factor_1}**, **{Factor_2}** y su interacción sobre **{Variable_y}**")

    #Modelo con interacción: y ~ A + B + A:B
    modelo = ols(f"{Variable_y} ~ C({Factor_1}) * C({Factor_2})", data=df_f).fit()
    tabla_anova = sm.stats.anova_lm(modelo, typ=1)

    #Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)

    # Contenedor A: tabla ANOVA e interpretación
    with Contenedor_A:
        st.subheader("Tabla ANOVA")
        st.dataframe(tabla_anova.round(4))
        for efecto, p in tabla_anova['PR(>F)'].dropna().items():
            if p < 0.05:
                st.success(f"✅ {efecto}: significativo (p = {p:.4f})")
            else:
                st.warning(f"⚠️ {efecto}: no significativo (p = {p:.4f})")

    # Contenedor B: gráfica de interacción
    with Contenedor_B:
        st.subheader("Gráfica de interacción")
        medias = df_f.groupby([Factor_1, Factor_2])[Variable_y].mean().reset_index()
        medias[Factor_2] = medias[Factor_2].astype(str)
        fig_int = px.line(medias, x=Factor_1, y=Variable_y, color=Factor_2, markers=True,
                          labels={Variable_y: f"Media de {Variable_y}"})
        st.plotly_chart(fig_int)
        st.caption("Si las líneas se cruzan o no son paralelas, hay interacción entre los factores.")

    #Fila 2: Heatmap de medias
    st.subheader("Heatmap de medias")
    tabla_medias = df_f.pivot_table(index=Factor_1, columns=Factor_2, values=Variable_y, aggfunc='mean')
    fig_hm, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(tabla_medias, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig_hm)
