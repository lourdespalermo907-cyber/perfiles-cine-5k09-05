"""Perfiles cinematográficos · Grupo 5K09-05 · UTN Ciencia de Datos.

Aplicación narrativa + exploración Altair + prueba del modelo reproducible.
Ejecutar: python -m streamlit run app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from modelado import (
    COLUMNAS_DECADAS, MIN_RATINGS, entrenar_desde_csv,
    construir_fila_respuestas, predecir_provisional, decada_columna,
)
from peliculas import PELICULAS
from recomendaciones import (
    RECOMENDACIONES, RECOMENDACIONES_K5, NOMBRES_CORTOS, G_AV, G_DR, D_MOD, D_ANT,
)

st.set_page_config(page_title='Perfiles de cine | 5K09-05', page_icon='🎬', layout='wide')
BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / 'usuarios_generos.csv'
alt.data_transformers.disable_max_rows()

COLORES = {
    (G_AV,D_MOD): '#277DA1',
    (G_AV,D_ANT): '#43AA8B',
    (G_DR,D_MOD): '#F8961E',
    (G_DR,D_ANT): '#B56576',
}
NOMBRE_DECADA = {
    'prop_antes_1980': 'Antes de 1980', 'prop_1980s':'Años 80',
    'prop_1990s':'Años 90', 'prop_2000s':'Años 2000',
    'prop_2010s':'Años 2010', 'prop_2020_mas':'2020 en adelante',
}
ORDEN_PERFILES = [(G_AV,D_MOD),(G_AV,D_ANT),(G_DR,D_MOD),(G_DR,D_ANT)]
PALETA = alt.Scale(domain=[NOMBRES_CORTOS[k] for k in ORDEN_PERFILES],
                   range=[COLORES[k] for k in ORDEN_PERFILES])

st.markdown('''<style>
.block-container {padding-top: 1.7rem; max-width: 1250px;}
h1,h2,h3 {letter-spacing:-0.025em;}
[data-testid="stMetricValue"]{font-weight:700;}
 .small-note {color:#5c6773;font-size:.94rem}
/* Navegación lateral de texto (sin círculos ni barras de radio). */
[class*="st-key-ir_seccion_"] button {
    width: 100%; justify-content: flex-start;
    text-align: left; border: none; box-shadow: none;
    background: transparent; padding: 0.58rem 0.7rem;
    color: inherit; font-weight: 450; border-radius: 0.35rem;
}
[class*="st-key-ir_seccion_"] button:hover {
    background: rgba(39,125,161,.10); color: #277DA1;
    text-decoration: underline;
}
/* Indicador de sección activa, sin formato de botón prominente. */
[class*="st-key-ir_seccion_"] button[kind="primary"] {
    background: rgba(39,125,161,.10);
    color: #277DA1; font-weight: 700;
}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Cargando datos y reproduciendo PCA y K-means…')
def cargar_modelos(ruta, marca):
    return entrenar_desde_csv(str(ruta))

@st.cache_data
def leer_bruto(ruta, marca):
    return pd.read_csv(ruta)


def filas_perfiles(modelos):
    datos = modelos.etiquetas.reset_index().merge(
        modelos.entrenamiento.reset_index()[['userId','cantidad_valoraciones',
                                           'rating_promedio_general','cantidad_valoraciones_total']],
        on='userId', validate='one_to_one')
    datos['perfil_corto'] = [NOMBRES_CORTOS[(g,d)] for g,d in zip(datos.nombre_genero,datos.nombre_decada)]
    return datos


def color_por_perfil(nombre):
    for clave, valor in NOMBRES_CORTOS.items():
        if valor == nombre:
            return COLORES[clave]
    return '#277DA1'


def grafico_scatter(datos, bloque='generos', marcador=None, alto=380):
    x,y = ('G_PC1','G_PC2') if bloque=='generos' else ('D_PC1','D_PC2')
    g = alt.Chart(datos).mark_circle(size=52, opacity=0.48).encode(
        x=alt.X(f'{x}:Q', title='Componente principal 1', scale=alt.Scale(zero=False)),
        y=alt.Y(f'{y}:Q', title='Componente principal 2', scale=alt.Scale(zero=False)),
        color=alt.Color('perfil_corto:N', title='Perfil combinado', scale=PALETA),
        tooltip=[alt.Tooltip('userId:O',title='Usuario'),
                 alt.Tooltip('perfil_corto:N',title='Perfil'),
                 alt.Tooltip('cantidad_valoraciones:Q',title='Ratings procesados'),
                 alt.Tooltip('rating_promedio_general:Q',title='Media ratings', format='.2f')],
    ).properties(height=alto)
    if marcador is not None:
        z = marcador['z_genero'] if bloque=='generos' else marcador['z_decada']
        punto = pd.DataFrame([{x:float(z[0]), y:float(z[1]),'detalle':'Caso seleccionado'}])
        punto_chart = alt.Chart(punto).mark_point(shape='diamond',filled=True,
                        size=310,color='#D62728',stroke='#ffffff',strokeWidth=2).encode(
            x=f'{x}:Q', y=f'{y}:Q', tooltip=['detalle:N'])
        return (g + punto_chart).interactive()
    return g.interactive()


def tabla_conteos(modelos):
    z = modelos.etiquetas.groupby(['nombre_genero','nombre_decada']).size().rename('Usuarios').reset_index()
    z['Perfil'] = [NOMBRES_CORTOS[(g,d)] for g,d in zip(z.nombre_genero,z.nombre_decada)]
    z['Porcentaje'] = 100*z['Usuarios']/len(modelos.etiquetas)
    return z


def ficha_recomendaciones(ng,nd, titulo=True):
    clave=(ng,nd)
    if titulo:
        st.subheader('Cinco películas para explorar este perfil')
    st.caption('Selección editorial de géneros y época compatibles; no es un ranking aprendido ni una predicción de puntuaciones.')
    for i,p in enumerate(RECOMENDACIONES[clave],1):
        st.markdown(f'**{i}. {p["titulo"]} ({p["anio"]})** · {p["generos"]}')
        st.caption(p['motivo'])
    st.caption('Los títulos sugeridos son distintos de los diez utilizados en el cuestionario.')


def distribucion_por_decada(fila):
    t=pd.DataFrame({'Década':[NOMBRE_DECADA[x] for x in COLUMNAS_DECADAS],
                    'Porcentaje':[100*float(fila[x].iloc[0]) for x in COLUMNAS_DECADAS]})
    return alt.Chart(t).mark_bar(color='#3B82A0').encode(
        x=alt.X('Década:N',sort=None,title=None,axis=alt.Axis(labelAngle=-30)),
        y=alt.Y('Porcentaje:Q',title='Porcentaje de películas calificadas'),
        tooltip=['Década:N',alt.Tooltip('Porcentaje:Q',format='.1f')],
    ).properties(height=240)


def preferencias_por_genero(modelos,fila):
    trans = modelos.generos['preparar']['features']
    t=trans.transform(fila).iloc[0].sort_values(ascending=False)
    df=pd.DataFrame({'Género':[x.replace('pref_','') for x in t.index],
                     'Preferencia relativa suavizada':t.values})
    return alt.Chart(df.head(12)).mark_bar(color='#3B82A0').encode(
        y=alt.Y('Género:N',sort='-x',title=None),
        x=alt.X('Preferencia relativa suavizada:Q',title='Respecto al promedio individual'),
        tooltip=['Género:N',alt.Tooltip('Preferencia relativa suavizada:Q',format='.3f')],
    ).properties(height=315)


def etiqueta_explicita(resultado, provisional=False):
    breve = NOMBRES_CORTOS[(resultado['nombre_genero'],resultado['nombre_decada'])]
    st.markdown('### Tu perfil cinematográfico ' + ('estimado' if provisional else 'asignado'))
    st.success('**' + breve + '**')
    st.write('**Géneros característicos:** ' + resultado['nombre_genero'] + '.')
    st.write('**Época más representada en el grupo:** ' + resultado['nombre_decada'] + '.')
    st.caption('Es una categoría descriptiva asignada por K-means, no una identidad fija ni una probabilidad.')
    return breve


def inicio(modelos,bruto):
    st.title('¿Qué nos dicen las películas que calificamos?')
    st.markdown('**Pregunta:** ¿Podemos identificar perfiles de usuarios a partir de sus patrones de valoración de películas y utilizarlos para orientar recomendaciones?')
    st.write('**Hallazgo:** los perfiles por géneros y por épocas de las películas calificadas capturan dos dimensiones diferentes. Al cruzarlas obtenemos cuatro perfiles descriptivos.')
    a,b,c=st.columns(3)
    a.metric('Usuarios de partida',f'{len(bruto):,}'.replace(',','.'))
    b.metric('Usuarios analizados',f'{len(modelos.entrenamiento):,}'.replace(',','.'))
    c.metric('Perfiles combinados',4)
    st.caption('MovieLens (valoraciones de usuarios) + TMDB (metadatos en el pipeline) · Ciencia de Datos · UTN · Grupo 5K09-05')
    st.divider()
    st.header('Los datos')
    st.subheader('Un mínimo de 150 valoraciones conserva aproximadamente el 70 % de los usuarios')
    umbrales=[100,150,200,250,300,400]
    resumen=pd.DataFrame({'Mínimo de valoraciones':umbrales,
                          'Usuarios':[int((bruto.cantidad_valoraciones>=x).sum()) for x in umbrales]})
    chart=alt.Chart(resumen).mark_bar(color='#277DA1').encode(
        x=alt.X('Mínimo de valoraciones:O',title='Mínimo X'),
        y=alt.Y('Usuarios:Q',title='Usuarios conservados',scale=alt.Scale(domain=[0,5000])),
        tooltip=['Mínimo de valoraciones:O','Usuarios:Q']).properties(height=250)
    st.altair_chart(chart,use_container_width=True)
    st.write('La media de valoraciones procesadas por usuario es **256,7** y la mediana **206**. Con X=150 se conservan **3.503 de 5.000** usuarios; elevar X descartaría muchos más historiales.')
    with st.expander('Cómo lo sabemos · obtención y limpieza', expanded=False):
        st.markdown('''**Procedencia:** MovieLens aporta ratings por usuario y película; TMDB aporta los metadatos de películas utilizados durante la construcción del dataset mediante Airflow.

**Unidad de análisis:** una fila = un usuario, con promedio general, promedios y cantidades por género y proporciones de películas calificadas por década de estreno.

**Filtrado:** hay 5.000 filas, sin duplicados de usuario. Se aplica `cantidad_valoraciones >= 150`, quedan 3.503 (70,06 %); se descartan 1.497. La columna `cantidad_valoraciones` está limitada a 500 en este CSV, por lo que no representa siempre el historial completo; `cantidad_valoraciones_total` refleja su recuento original.

**Control de fuga:** no utilizamos ID de usuario, identificadores de grupo, `genero_mejor_valorado` ni variables que ya contengan la etiqueta objetivo. El modelado es no supervisado.''')
    st.divider()
    st.header('Lo que encontramos')
    st.subheader('Los géneros y las décadas aportan dos descripciones complementarias')
    datos = tabla_conteos(modelos)
    # Etiquetas breves: los nombres completos se mantienen en el tooltip.
    datos['Géneros'] = datos['nombre_genero'].map({
        G_AV: 'Animación y aventura',
        G_DR: 'Drama y crimen',
    })
    datos['Época'] = datos['nombre_decada'].map({
        D_MOD: '2000–2010',
        D_ANT: 'Clásicos y 90s',
    })
    datos['Porcentaje visible'] = datos['Porcentaje'].map(lambda p: f'{p:.1f} %'.replace('.', ','))
    eje_x = alt.X('Época:N', sort=['2000–2010', 'Clásicos y 90s'],
                  axis=alt.Axis(labelAngle=0, labelFontSize=13, labelLimit=220),
                  title='Época de estreno de las películas calificadas')
    eje_y = alt.Y('Géneros:N', sort=['Animación y aventura', 'Drama y crimen'],
                  axis=alt.Axis(labelFontSize=13, labelLimit=210),
                  title='Perfil de géneros')
    posiciones = dict(x=eje_x, y=eje_y)
    base = alt.Chart(datos).encode(**posiciones)
    rectangulos = base.mark_rect(stroke='white', strokeWidth=3).encode(
        color=alt.Color('Usuarios:Q', scale=alt.Scale(scheme='blues'), legend=None),
        tooltip=[alt.Tooltip('Perfil:N', title='Perfil combinado'),
                 alt.Tooltip('Usuarios:Q', title='Usuarios'),
                 alt.Tooltip('Porcentaje:Q', title='Porcentaje del total', format='.1f')],
    )
    cifras = base.mark_text(fontSize=22, fontWeight='bold', dy=-10).encode(
        text=alt.Text('Usuarios:Q', format=',.0f'),
        color=alt.condition('datum.Usuarios > 900', alt.value('white'), alt.value('#16324f')),
    )
    porcentajes = base.mark_text(fontSize=13, dy=18).encode(
        text='Porcentaje visible:N',
        color=alt.condition('datum.Usuarios > 900', alt.value('white'), alt.value('#16324f')),
    )
    matriz = (rectangulos + cifras + porcentajes).properties(height=265)
    st.altair_chart(matriz, use_container_width=True)
    st.caption('Cada casilla representa una combinación de perfil de géneros y época. '
               'Los valores indican usuarios y porcentaje del total; el cursor muestra el nombre completo.')
    st.write('En los dos grupos de géneros, aproximadamente la mitad de los usuarios cae en cada grupo de décadas. Esto sugiere que el patrón de géneros y la época de las películas calificadas son dimensiones distintas.')
    with st.expander('Cómo lo sabemos · hipótesis y transformaciones',expanded=False):
        st.markdown('''**Hipótesis sostenida:** la combinación de géneros y décadas aporta perfiles descriptivos más detallados que una sola etiqueta. La tabla muestra cuatro intersecciones no vacías.

**Hipótesis que no se sostuvo:** que quienes tienen un perfil de género determinado concentren su historial en una época particular. Los porcentajes de D0 y D1 son prácticamente iguales en ambos grupos de géneros (49,7/50,3 % y 49,8/50,2 %); prueba de asociación χ²: `p≈0,9915` y V de Cramér ≈0.

**Cómo se calculó:** se agruparon independientemente los scores relativos suavizados de géneros y las proporciones de películas calificadas por década. Las celdas muestran conteos de usuarios, NO cantidad de películas ni afinidad de 1 a 5 por época.''')
    st.divider()
    st.header('El modelo')
    st.subheader('Dos K-means separados conservan información que el modelo conjunto perdía')
    metricas=pd.DataFrame({'Modelo':['Géneros (K=2)','Décadas (K=2)','Conjunto (K=2)','Géneros temáticos (K=5)'],
                           'Silhouette':[.180,.424,.224,.102],
                           'Decisión':['Elegido','Elegido','Alternativa','Exploratorio']})
    bar=alt.Chart(metricas).mark_bar(color='#277DA1').encode(
        x=alt.X('Silhouette:Q',scale=alt.Scale(domain=[0,.5]),title='Silhouette (en el espacio de cada modelo)'),
        y=alt.Y('Modelo:N',sort='-x',title=None),
        tooltip=['Modelo:N',alt.Tooltip('Silhouette:Q',format='.3f'),'Decisión:N']
    ).properties(height=215)
    st.altair_chart(bar,use_container_width=True)
    st.write('Se eligió **K=2 en géneros y K=2 en décadas**, y se cruzaron ambas etiquetas: **4 perfiles combinados**. El modelo conjunto tuvo una partición muy parecida a la temporal y añadió poca diferenciación de géneros. K=5 ofrece perfiles temáticos más específicos, pero con menor separación.')
    st.caption('Los silhouette de bloques con variables distintas no equivalen a una medida de calidad de recomendaciones.')
    st.dataframe(datos[['Perfil','Usuarios','Porcentaje']].style.format({'Porcentaje':'{:.1f} %'}),hide_index=True,use_container_width=True)
    with st.expander('Cómo lo sabemos · selección, métricas y estabilidad',expanded=False):
        st.markdown('''**Variables:** scores por género centrados en la media de rating del usuario y suavizados con `n/(n+15)`; se exige cobertura ≥75 % para elegir un género. Para décadas: proporciones de películas calificadas, transformadas con `arcsin(sqrt(p))` y con 2020+ excluido del espacio de distancia por escasa cobertura.

**Algoritmo:** `StandardScaler → PCA` (retener ≥80 % de varianza) `→ KMeans`, entrenados con semilla `42` y `n_init=10` en cada bloque. Se conservaron 8 PCs de géneros y 3 de décadas. PC1/PC2 se muestran para explorar; K-means usa todas las componentes retenidas.

**Selección:** k de 2 a 6, usando curvas de inercia, silhouette y tamaños. K=2 da silhouette 0,180 para géneros y 0,424 para décadas. K=5 para géneros da 0,102, aunque ofrece interpretación temática más detallada.

**Estabilidad:** al reentrenar sobre cuatro submuestras del 80 % se midió ARI promedio 0,968 (géneros) y 0,990 (décadas). No se trata de precisión predictiva.

**Alternativa conjunta:** silhouette 0,224; ARI con décadas ≈0,922 y con géneros ≈0. La decisión es conservar ambos modelos por separado y cruzarlos.

**Romance:** sacarlo mejoró marginalmente silhouette a 0,1847 con ARI ≈0,981 respecto del modelo completo. Se mantiene por coherencia; no se crean categorías artificiales Comedy-Romance.

**Límite:** no hay etiquetas verdaderas de tipo de espectador y no evaluamos todavía exactitud de recomendaciones. Las asignaciones de K-means no tienen probabilidades de pertenencia.''')


def exploracion(modelos):
    st.header('Explorá los perfiles de los usuarios')
    st.write('Cada punto representa un usuario. Aplicá filtros y elegí un caso para ver sus características y el grupo que le asigna el modelo.')
    todo=filas_perfiles(modelos)
    c1,c2,c3=st.columns([1,1,1])
    with c1:
        bloque=st.selectbox('Proyección PCA',['Géneros','Décadas'],key='ex_bloque')
    with c2:
        seleccionados=st.multiselect('Perfiles combinados',options=[NOMBRES_CORTOS[k] for k in ORDEN_PERFILES],
                                     default=[NOMBRES_CORTOS[k] for k in ORDEN_PERFILES],key='ex_filtros')
    with c3:
        minimo=st.slider('Mínimo de valoraciones procesadas',150,500,150,step=25,key='ex_minimo')
    datos=todo[todo.perfil_corto.isin(seleccionados) & (todo.cantidad_valoraciones>=minimo)].copy()
    st.caption(f'{len(datos):,} usuarios seleccionados de {len(todo):,}. Usá zoom y desplazamiento sobre el gráfico.'.replace(',','.'))
    if datos.empty:
        st.warning('Los filtros no dejaron usuarios. Cambiá el mínimo o seleccioná un perfil.')
        return
    st.altair_chart(grafico_scatter(datos,'generos' if bloque=='Géneros' else 'decadas',alto=450),use_container_width=True)
    ids=datos['userId'].tolist()
    elegido=st.selectbox('Ver un usuario en detalle',ids,format_func=lambda u:f'Usuario {u}',key='ex_usuario')
    fila=modelos.entrenamiento.loc[[elegido]]
    pred=predecir_provisional(modelos,fila)
    etiqueta_explicita(pred,provisional=False)
    a,b,c=st.columns(3)
    a.metric('Valoraciones procesadas',int(fila['cantidad_valoraciones'].iloc[0]))
    b.metric('Valoraciones originales',int(fila['cantidad_valoraciones_total'].iloc[0]))
    c.metric('Promedio individual',f'{float(fila["rating_promedio_general"].iloc[0]):.2f} / 5')
    g,d=st.columns(2)
    with g:
        st.markdown('**Preferencias relativas por género**')
        st.altair_chart(preferencias_por_genero(modelos,fila),use_container_width=True)
    with d:
        st.markdown('**Películas calificadas según década de estreno**')
        st.altair_chart(distribucion_por_decada(fila),use_container_width=True)
    with st.expander('Cinco sugerencias para este perfil',expanded=False):
        ficha_recomendaciones(pred['nombre_genero'],pred['nombre_decada'],titulo=False)


def datos_del_cuestionario(respuestas):
    g,d={},{}
    for peli in PELICULAS:
        r=respuestas.get(peli['id'])
        if not isinstance(r,int) or isinstance(r,bool):
            continue
        for gen in peli['generos']:
            g.setdefault(gen,[]).append(r)
        d.setdefault(NOMBRE_DECADA[decada_columna(peli['anio'])],[]).append(r)
    df_g=pd.DataFrame([{'Género':k,'Promedio':np.mean(v),'Cantidad':len(v)} for k,v in g.items()])
    df_d=pd.DataFrame([{'Década':k,'Promedio':np.mean(v),'Cantidad':len(v)} for k,v in d.items()])
    return df_g,df_d


def mostrar_respuestas(respuestas):
    g,d=datos_del_cuestionario(respuestas)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('**Puntuaciones del cuestionario por género**')
        st.altair_chart(alt.Chart(g).mark_bar(color='#277DA1').encode(
            y=alt.Y('Género:N',sort='-x',title=None),
            x=alt.X('Promedio:Q',title='Media de 1 a 5',scale=alt.Scale(domain=[0,5])),
            tooltip=['Género:N',alt.Tooltip('Promedio:Q',format='.2f'),'Cantidad:Q']
        ).properties(height=max(250,22*len(g))),use_container_width=True)
    with c2:
        st.markdown('**Puntuaciones del cuestionario por década**')
        st.altair_chart(alt.Chart(d).mark_bar(color='#F8961E').encode(
            y=alt.Y('Década:N',title=None),
            x=alt.X('Promedio:Q',title='Media de 1 a 5',scale=alt.Scale(domain=[0,5])),
            tooltip=['Década:N',alt.Tooltip('Promedio:Q',format='.2f'),'Cantidad:Q']
        ).properties(height=260),use_container_width=True)
    st.caption('Son descripciones de tus respuestas. El modelo de décadas usa proporciones de películas calificadas, NO el rating medio por década.')


def prueba(modelos):
    st.header('Probá el modelo: descubrí tu perfil cinematográfico')
    st.write('Calificá las diez películas del 1 al 5. Si no viste una, elegí **«No la vi»**: no cuenta como cero.')
    st.write('Necesitás al menos **seis valoraciones válidas** para obtener una **estimación experimental**.')
    st.caption('Las diez películas cubren géneros distintos y cinco fueron estrenadas antes de 2000 y cinco a partir de 2000. Solo una incluye Aventura.')
    respuestas={}
    with st.form('form_cine'):
        for i,peli in enumerate(PELICULAS,1):
            a,b=st.columns([3,2],vertical_alignment='center')
            with a:
                st.markdown(f'**{i}. {peli["titulo"]} ({peli["anio"]})**')
                st.caption(' · '.join(peli['generos']))
            with b:
                opciones=['No la vi',1,2,3,4,5]
                respuestas[peli['id']]=st.selectbox('Rating: '+peli['titulo'],opciones,
                    index=None,placeholder='Elegí una opción',
                    format_func=lambda x: 'No la vi' if x == 'No la vi' else ('★' * x + '☆' * (5 - x)),
                    label_visibility='collapsed',key='quiz_'+peli['id'])
        listo=st.form_submit_button('Descubrir mi perfil',type='primary')
    if listo:
        if any(x is None for x in respuestas.values()):
            st.error('Respondé las diez películas. Si no conocés alguna, marcá «No la vi».')
            st.session_state.pop('ultimo_quiz',None)
        else:
            val={k:('no_vi' if v=='No la vi' else v) for k,v in respuestas.items()}
            vistas=sum(isinstance(v,int) and not isinstance(v,bool) for v in val.values())
            if vistas<6:
                st.warning(f'Calificaste {vistas} películas. Necesitamos al menos seis para una estimación provisional.')
                st.session_state.pop('ultimo_quiz',None)
            else:
                fila,estadisticas=construir_fila_respuestas(val,PELICULAS,list(modelos.entrenamiento.columns))
                pred=predecir_provisional(modelos,fila)
                st.session_state['ultimo_quiz']=(val,estadisticas,pred)
    guardado=st.session_state.get('ultimo_quiz')
    if guardado is None:
        st.info('El perfil y sus cinco recomendaciones aparecerán después de responder el cuestionario.')
        return
    val,estadisticas,pred=guardado
    st.divider()
    etiqueta_explicita(pred,provisional=True)
    a,b,c=st.columns(3)
    a.metric('Películas puntuadas',estadisticas['n_vistas'])
    b.metric('Géneros cubiertos',estadisticas['n_generos_cubiertos'])
    c.metric('Décadas cubiertas',estadisticas['n_decadas_cubiertas'])
    ficha_recomendaciones(pred['nombre_genero'],pred['nombre_decada'])
    st.warning('**Estimación de baja evidencia:** el modelo se entrenó con historiales de al menos 150 valoraciones, mientras que este formulario tiene como máximo diez. El resultado puede ser inestable y no se informa probabilidad de acierto.')
    if pred['fuera_rango_genero'] or pred['fuera_rango_decada']:
        st.warning('Tu caso está más lejos de un centro que el 95 % de los casos de entrenamiento en al menos uno de los bloques. El perfil es especialmente incierto.')
    if estadisticas['n_decadas_cubiertas']<3:
        st.warning('Se calificaron películas de menos de tres épocas; el perfil de décadas está poco representado.')
    st.caption('El grupo por décadas depende de qué épocas están presentes entre las películas puntuadas, no de si te gustaron más o menos.')
    mostrar_respuestas(val)
    historicos=filas_perfiles(modelos)
    g,d=st.columns(2)
    with g:
        st.markdown('**Tu ubicación en el PCA de géneros**')
        st.altair_chart(grafico_scatter(historicos,'generos',pred),use_container_width=True)
    with d:
        st.markdown('**Tu ubicación en el PCA de décadas**')
        st.altair_chart(grafico_scatter(historicos,'decadas',pred),use_container_width=True)
    with st.expander('Cómo lo sabemos · conversión de respuestas y límites',expanded=False):
        st.markdown('''Para cada película puntuada se recuperan sus géneros y año de estreno. Las respuestas «No la vi» se excluyen completamente. Se generan promedios y cantidades por género, rating general y proporciones por década de estreno. Después se aplican las **transformaciones, StandardScaler, PCA y K-means entrenados con el CSV real**.

Como el suavizado de géneros es `n/(n+15)`, con una sola película de un género la información queda muy atenuada. Por otro lado, una selección fija de diez películas **no permite identificar de manera fiable** una preferencia real por épocas: si dos personas puntuaron los mismos títulos, su bloque temporal será idéntico aunque hayan dado notas opuestas. El cuestionario es una demostración, no una validación estadística de un recomendador.

Las distancias geométricas a centroides **no equivalen a probabilidades**. En particular, no se entrenó ningún modelo supervisado para predecir gustos personales.''')
        st.dataframe(pd.DataFrame({'Bloque':['Géneros','Décadas'],
                      'Distancia a centro':[pred['distancia_genero'],pred['distancia_decada']],
                      'P95 de entrenamiento':[modelos.distancias_p95['generos'],modelos.distancias_p95['decadas']]}),
                      hide_index=True,use_container_width=True)


def catalogo():
    st.header('Las cinco películas propuestas para cada tipo de usuario')
    st.write('Son 20 títulos distintos de los diez del cuestionario. Se eligieron por adecuación temática y fecha de estreno al perfil, **no por un ranking calculado con ratings individuales**.')
    for ng,nd in ORDEN_PERFILES:
        with st.expander(NOMBRES_CORTOS[(ng,nd)]+' — 5 películas',expanded=False):
            ficha_recomendaciones(ng,nd,titulo=False)
    st.divider()
    st.subheader('Alternativa exploratoria: los cinco perfiles temáticos de K=5')
    st.write('Se evaluó también K=5 en géneros. Aparecen tendencias temáticas más específicas, aunque el silhouette baja de 0,180 a 0,102. **El cuestionario utiliza el modelo principal K=2.** Estas películas ilustran las cinco tendencias alternativas.')
    for nombre, opciones in RECOMENDACIONES_K5.items():
        with st.expander(nombre + ' — 5 películas', expanded=False):
            for i,(titulo,anio,generos) in enumerate(opciones,1):
                st.markdown(f'**{i}. {titulo} ({anio})** · {generos}')
            st.caption('Sugerencias curatoriales de una tendencia exploratoria, no recomendaciones validadas ni asignaciones del modelo principal.')
    with st.expander('Cómo se eligieron las películas', expanded=False):
        st.write('Las sugerencias se seleccionaron por afinidad temática y época de estreno. '
                 'No son predicciones de cuánto puntuaría una persona cada película. '
                 'Validar un recomendador requeriría interacciones individuales no vistas '
                 'y métricas como precisión@K y recall@K.')


def pie():
    st.divider()
    st.caption('Grupo 5K09-05 · Ciencia de Datos · UTN · Fuente de datos: MovieLens / TMDB · Modelado: PCA + K-means. Clasificaciones descriptivas, no etiquetas verdaderas.')


def abrir_seccion(nombre):
    """El enlace lateral selecciona la misma pestaña que se ve arriba."""
    st.session_state['pestana_activa'] = nombre


def main():
    if not ARCHIVO.exists():
        st.error('Falta usuarios_generos.csv en la misma carpeta que app.py.')
        st.stop()
    marca=ARCHIVO.stat().st_mtime_ns
    modelos=cargar_modelos(str(ARCHIVO),marca)
    bruto=leer_bruto(str(ARCHIVO),marca)
    # Pestañas superiores y enlaces laterales sincronizados por Session State.
    # st.tabs(key=..., on_change="rerun") permite cambiar pestañas desde botones.
    titulos = ['El proyecto', 'Explorá', 'Probá el modelo', 'Películas por perfil']
    enlaces = ['El proyecto y sus hallazgos', 'Explorá los perfiles',
               'Probá el cuestionario', 'Películas por perfil']
    with st.sidebar:
        st.markdown('## Perfiles de cine')
        st.caption('Ciencia de Datos · Grupo 5K09-05')
        st.markdown('**Cómo recorrer la aplicación**')
        for i, (destino, nombre) in enumerate(zip(titulos, enlaces)):
            activo = st.session_state.get('pestana_activa', titulos[0]) == destino
            st.button(('▸ ' if activo else '   ') + nombre, key=f'ir_seccion_{i}', use_container_width=True,
                      type='tertiary',
                      on_click=abrir_seccion, args=(destino,))
        st.info('El modelo principal combina dos grupos de géneros y dos grupos de décadas.')
        st.caption('Los géneros describen afinidades relativas; las épocas indican cuándo se estrenaron las películas calificadas.')

    paneles = st.tabs(titulos, key='pestana_activa', on_change='rerun')
    if paneles[0].open:
        with paneles[0]:
            inicio(modelos, bruto)
    if paneles[1].open:
        with paneles[1]:
            exploracion(modelos)
    if paneles[2].open:
        with paneles[2]:
            prueba(modelos)
    if paneles[3].open:
        with paneles[3]:
            catalogo()
    pie()

if __name__=='__main__':
    main()
