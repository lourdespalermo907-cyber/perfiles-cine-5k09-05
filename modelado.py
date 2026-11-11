"""Inferencia reproducible de la Entrega 3 del grupo 5K09-05.

Criterio oficial conservado: 150 ratings, m=15, cobertura=.75,
PCA 80 %, arcsin(sqrt(proporción)) para décadas, K=2 en ambos bloques.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
MIN_RATINGS = 150
SUAVIZADO = 15
COBERTURA = 0.75
VARIANZA_PCA = 0.80

COLUMNAS_DECADAS = [
    "prop_antes_1980", "prop_1980s", "prop_1990s", "prop_2000s",
    "prop_2010s", "prop_2020_mas",
]
NOMBRES_GRUPOS = {
    "generos": {0: "Animación, familia y aventura", 1: "Drama, crimen e histórico"},
    "decadas": {0: "Cine de 2000–2010", 1: "Cine noventero y clásico"},
}


class PreferenciasGeneros(BaseEstimator, TransformerMixin):
    """Reproduce exactamente los scores relativos suavizados de Entrega 3."""

    def __init__(self, m=15, cobertura_min=0.75, excluir=()):
        self.m = m
        self.cobertura_min = cobertura_min
        self.excluir = excluir

    def fit(self, X, y=None):
        candidatos = [c[2:] for c in X.columns
                      if c.startswith('n_') and c[2:] not in self.excluir]
        self.generos_ = [g for g in candidatos
                         if (X[f'n_{g}'].fillna(0) > 0).mean() >= self.cobertura_min]
        if len(self.generos_) < 2:
            raise ValueError('Menos de dos géneros con cobertura suficiente')
        return self

    def transform(self, X):
        general = pd.to_numeric(X['rating_promedio_general'], errors='coerce')
        if general.isna().any():
            raise ValueError('rating_promedio_general nulo')
        salida = {}
        for g in self.generos_:
            n = pd.to_numeric(X[f'n_{g}'], errors='coerce').fillna(0).clip(lower=0)
            rating = pd.to_numeric(X[g], errors='coerce').fillna(general)
            salida[f'pref_{g}'] = (n / (n + self.m)) * (rating - general)
        return pd.DataFrame(salida, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'pref_{g}' for g in self.generos_])


class ProporcionesDecadas(BaseEstimator, TransformerMixin):
    """Fracción de títulos valorados por época; NO mide gusto por esa década."""

    def __init__(self, incluir_2020=False, transformacion='arcsin'):
        self.incluir_2020 = incluir_2020
        self.transformacion = transformacion

    def fit(self, X, y=None):
        self.columnas_ = [c for c in COLUMNAS_DECADAS
                          if c != 'prop_2020_mas' or self.incluir_2020]
        if self.transformacion not in ('arcsin', 'sin_transformar'):
            raise ValueError('Transformación de décadas desconocida')
        return self

    def transform(self, X):
        prop = X[self.columnas_].astype(float).clip(0, 1)
        if prop.isna().any().any():
            raise ValueError('Proporciones por década nulas')
        if self.transformacion == 'arcsin':
            return pd.DataFrame(np.arcsin(np.sqrt(prop.values)),
                                columns=self.columnas_, index=X.index)
        return prop.copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columnas_)


def preparar_generos():
    return Pipeline([
        ('features', PreferenciasGeneros(m=SUAVIZADO, cobertura_min=COBERTURA, excluir=())),
        ('escalar', StandardScaler()),
        ('pca', PCA(n_components=VARIANZA_PCA, svd_solver='full', random_state=SEMILLA)),
    ])


def preparar_decadas():
    return Pipeline([
        ('features', ProporcionesDecadas(incluir_2020=False, transformacion='arcsin')),
        ('escalar', StandardScaler()),
        ('pca', PCA(n_components=VARIANZA_PCA, svd_solver='full', random_state=SEMILLA)),
    ])


@dataclass
class Modelos:
    generos: Pipeline
    decadas: Pipeline
    historicos: pd.DataFrame
    etiquetas: pd.DataFrame
    entrenamiento: pd.DataFrame
    distancias_p95: dict


def entrenar_desde_csv(ruta: str) -> Modelos:
    """Entrena exactamente con el CSV y la semilla de la Entrega 3."""
    df = pd.read_csv(ruta)
    if df['userId'].duplicated().any():
        raise ValueError('Hay usuarios duplicados')
    if not set(COLUMNAS_DECADAS).issubset(df.columns):
        raise ValueError('Faltan proporciones de décadas')
    D = df.loc[df['cantidad_valoraciones'] >= MIN_RATINGS].set_index('userId').copy()
    if len(D) < 100:
        raise ValueError('Muestra de entrenamiento insuficiente')
    G = Pipeline([
        ('preparar', preparar_generos()),
        ('kmeans', KMeans(n_clusters=2, random_state=SEMILLA, n_init=10)),
    ]).fit(D)
    E = Pipeline([
        ('preparar', preparar_decadas()),
        ('kmeans', KMeans(n_clusters=2, random_state=SEMILLA, n_init=10)),
    ]).fit(D)
    gr = G.predict(D)
    de = E.predict(D)
    # Identificación dinámica basada en perfiles para evitar invertir etiquetas si se reentrena.
    pref = G['preparar']['features'].transform(D)
    medias = pref.groupby(gr).mean().T
    criterio_drama = (medias.loc['pref_Drama'] + medias.loc['pref_Crime']
                      + medias.loc['pref_History'] + medias.loc['pref_War']
                      - medias.loc['pref_Animation'] - medias.loc['pref_Family']
                      - medias.loc['pref_Adventure'] - medias.loc['pref_Fantasy'])
    id_drama = int(criterio_drama.idxmax())
    id_aventura = int(criterio_drama.idxmin())
    dp = (100 * D[COLUMNAS_DECADAS]).groupby(de).mean().T
    id_reciente = int((dp.loc['prop_2000s'] + dp.loc['prop_2010s']).idxmax())
    id_clasico = int((dp.loc['prop_2000s'] + dp.loc['prop_2010s']).idxmin())
    nombre_genero = {id_aventura: 'Animación, familia y aventura',
                     id_drama: 'Drama, crimen e histórico'}
    nombre_decada = {id_reciente: 'Cine de 2000–2010',
                     id_clasico: 'Cine noventero y clásico'}
    out = pd.DataFrame({
        'grupo_genero': gr, 'grupo_decada': de,
        'nombre_genero': [nombre_genero[i] for i in gr],
        'nombre_decada': [nombre_decada[i] for i in de],
    }, index=D.index)
    out['perfil'] = out['nombre_genero'] + ' · ' + out['nombre_decada']
    z_g = G['preparar'].transform(D)
    z_d = E['preparar'].transform(D)
    out['G_PC1'] = z_g[:, 0]
    out['G_PC2'] = z_g[:, 1]
    out['D_PC1'] = z_d[:, 0]
    out['D_PC2'] = z_d[:, 1]
    distancias = {}
    for bloque, modelo, Z, etiquetas in [
        ('generos', G, z_g, gr), ('decadas', E, z_d, de)
    ]:
        dist_asignada = modelo['kmeans'].transform(Z)[np.arange(len(Z)), etiquetas]
        distancias[bloque] = float(np.quantile(dist_asignada, .95))
    return Modelos(G, E, out, out, D, distancias)


def decada_columna(anio: int) -> str:
    if anio < 1980: return 'prop_antes_1980'
    if anio < 1990: return 'prop_1980s'
    if anio < 2000: return 'prop_1990s'
    if anio < 2010: return 'prop_2000s'
    if anio < 2020: return 'prop_2010s'
    return 'prop_2020_mas'


def construir_fila_respuestas(respuestas: dict, peliculas: list, columnas_entrada: list):
    """Convierte 10 respuestas a la estructura exacta del CSV de la Entrega 3.

    respuestas: {id_pelicula: int 1..5, 'no_vi' o None}.
    'No la vi' NO cuenta como valoración ni como rating 0.
    """
    validas = []
    for peli in peliculas:
        resp = respuestas.get(peli['id'])
        if resp is None or resp == 'no_vi':
            continue
        if not isinstance(resp, int) or isinstance(resp, bool) or resp < 1 or resp > 5:
            raise ValueError('Calificación fuera de la escala 1–5')
        validas.append((peli, float(resp)))
    if not validas:
        raise ValueError('No hay películas puntuadas')
    n = len(validas)
    prom = sum(r for _, r in validas) / n
    fila = {c: np.nan for c in columnas_entrada}
    fila['rating_promedio_general'] = prom
    fila['cantidad_valoraciones'] = n
    fila['cantidad_valoraciones_total'] = n
    generos = [c[2:] for c in columnas_entrada if c.startswith('n_')]
    for genero in generos:
        valoraciones = [rating for peli, rating in validas if genero in peli['generos']]
        fila[f'n_{genero}'] = len(valoraciones)
        fila[genero] = float(np.mean(valoraciones)) if valoraciones else np.nan
    for columna in COLUMNAS_DECADAS:
        fila[columna] = sum(decada_columna(p['anio']) == columna for p, _ in validas) / n
    detalle = pd.DataFrame([fila], index=pd.Index(['cuestionario'], name='userId'))
    datos = {
        'n_respondidas': sum(respuestas.get(p['id']) is not None for p in peliculas),
        'n_vistas': n,
        'n_generos_cubiertos': sum(fila.get('n_'+g, 0) > 0 for g in generos),
        'n_decadas_cubiertas': sum(fila[c] > 0 for c in COLUMNAS_DECADAS),
        'media_rating': prom,
    }
    return detalle, datos


def predecir_provisional(modelos: Modelos, fila: pd.DataFrame):
    mg, md = modelos.generos, modelos.decadas
    cg = int(mg.predict(fila)[0])
    cd = int(md.predict(fila)[0])
    zg = mg['preparar'].transform(fila)[0]
    zd = md['preparar'].transform(fila)[0]
    dg = float(mg['kmeans'].transform([zg])[0, cg])
    dd = float(md['kmeans'].transform([zd])[0, cd])
    nombre_g = modelos.etiquetas.loc[modelos.etiquetas.grupo_genero == cg, 'nombre_genero'].iloc[0]
    nombre_d = modelos.etiquetas.loc[modelos.etiquetas.grupo_decada == cd, 'nombre_decada'].iloc[0]
    return {
        'grupo_genero': cg, 'grupo_decada': cd,
        'nombre_genero': nombre_g, 'nombre_decada': nombre_d,
        'nombre_perfil': nombre_g + ' · ' + nombre_d,
        'z_genero': zg, 'z_decada': zd,
        'distancia_genero': dg, 'distancia_decada': dd,
        'fuera_rango_genero': dg > modelos.distancias_p95['generos'],
        'fuera_rango_decada': dd > modelos.distancias_p95['decadas'],
    }
