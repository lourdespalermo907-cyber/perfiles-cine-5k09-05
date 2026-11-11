"""Pruebas con el CSV de la Entrega 3: python -m pytest -q"""
from pathlib import Path
import pandas as pd
import pytest
from modelado import (
    entrenar_desde_csv, construir_fila_respuestas, predecir_provisional,
    COLUMNAS_DECADAS, MIN_RATINGS,
)
from peliculas import PELICULAS
from recomendaciones import (
    RECOMENDACIONES,RECOMENDACIONES_K5,validar_catalogo,validar_catalogo_k5,
)

BASE=Path(__file__).resolve().parent

@pytest.fixture(scope='module')
def modelos():
    return entrenar_desde_csv(str(BASE/'usuarios_generos.csv'))


def test_modelos_y_grupos_coinciden_entrega3(modelos):
    conteos=modelos.etiquetas.groupby(['grupo_genero','grupo_decada']).size().to_dict()
    assert len(modelos.entrenamiento)==3503
    assert conteos=={(0,0):1096,(0,1):1108,(1,0):647,(1,1):652}
    assert modelos.generos['kmeans'].n_clusters==2
    assert modelos.decadas['kmeans'].n_clusters==2


def test_usuario_historico_recibe_misma_etiqueta(modelos):
    for userid in modelos.etiquetas.index[:12]:
        r=predecir_provisional(modelos,modelos.entrenamiento.loc[[userid]])
        assert r['nombre_perfil']==modelos.etiquetas.loc[userid,'perfil']


def test_no_la_vi_no_impacta_en_ratings(modelos):
    r={p['id']:'no_vi' for p in PELICULAS}
    for i,p in enumerate(PELICULAS[:6]):
        r[p['id']]=i%5+1
    fila,d=construir_fila_respuestas(r,PELICULAS,list(modelos.entrenamiento.columns))
    assert d['n_vistas']==6
    assert d['n_respondidas']==10
    assert fila['cantidad_valoraciones'].iloc[0]==6
    assert abs(sum(float(fila[c].iloc[0]) for c in COLUMNAS_DECADAS)-1)<1e-10
    assert abs(float(fila['rating_promedio_general'].iloc[0])-sum(i%5+1 for i in range(6))/6)<1e-10


def test_deudas_de_decadas_sin_confundir_gusto(modelos):
    # Si las películas calificadas son las mismas, cambiar estrellas no cambia las proporciones de décadas.
    a={p['id']:1 for p in PELICULAS}
    b={p['id']:5 for p in PELICULAS}
    fa,_=construir_fila_respuestas(a,PELICULAS,list(modelos.entrenamiento.columns))
    fb,_=construir_fila_respuestas(b,PELICULAS,list(modelos.entrenamiento.columns))
    pd.testing.assert_frame_equal(fa[COLUMNAS_DECADAS],fb[COLUMNAS_DECADAS])
    assert predecir_provisional(modelos,fa)['grupo_decada']==predecir_provisional(modelos,fb)['grupo_decada']


def test_todos_los_perfiles_tienen_cinco_recomendaciones_no_repetidas():
    validar_catalogo()
    validar_catalogo_k5()
    assert sum(len(x) for x in RECOMENDACIONES.values())==20
    assert sum(len(x) for x in RECOMENDACIONES_K5.values())==25
    assert MIN_RATINGS==150
