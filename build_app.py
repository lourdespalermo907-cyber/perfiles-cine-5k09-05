"""Regenera salidas auditables a partir del CSV del pipeline de Airflow.

La propia app recalcula, en memoria y con caché, los mismos modelos reproducibles.
Ejecutar: python build_app.py
"""
from pathlib import Path
import json
import pandas as pd
from modelado import entrenar_desde_csv, MIN_RATINGS
from recomendaciones import validar_catalogo

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'usuarios_generos.csv'
DEST=ROOT/'resultados_reproducibles'


def main():
    validar_catalogo()
    DEST.mkdir(exist_ok=True)
    modelos=entrenar_desde_csv(str(BASE))
    etiquetados=modelos.etiquetas.reset_index()
    etiquetados.to_csv(DEST/'usuarios_perfiles.csv',index=False,encoding='utf-8-sig')
    perfiles=etiquetados.groupby(['nombre_genero','nombre_decada','perfil'],as_index=False).size()
    perfiles.rename(columns={'size':'usuarios'}).to_csv(DEST/'perfiles_resumen.csv',index=False,encoding='utf-8-sig')
    resumen={'grupo':'5K09-05','modelo':'PCA + K-means por géneros y décadas (k=2 cada uno)',
             'min_ratings':MIN_RATINGS,'usuarios_finales':int(len(etiquetados)),
             'perfiles':int(etiquetados['perfil'].nunique()),
             'importante':'Las décadas miden la distribución de películas calificadas, no gustos por época.'}
    (DEST/'metodologia.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Generación correcta:',len(etiquetados),'usuarios y',len(perfiles),'perfiles.')

if __name__=='__main__':
    main()
