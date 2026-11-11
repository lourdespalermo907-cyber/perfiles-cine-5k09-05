"""Revisión estructural del catálogo y la navegación."""
from collections import Counter
from pathlib import Path
from peliculas import PELICULAS
from recomendaciones import RECOMENDACIONES, RECOMENDACIONES_K5, validar_catalogo, validar_catalogo_k5


def test_cuestionario_variedad_y_periodos():
    assert len(PELICULAS) == 10
    assert len({p['id'] for p in PELICULAS}) == 10
    assert len({p['titulo'] for p in PELICULAS}) == 10
    generos = Counter(g for p in PELICULAS for g in p['generos'])
    assert generos['Adventure'] == 1
    assert sum(p['anio'] < 2000 for p in PELICULAS) == 5
    assert sum(p['anio'] >= 2000 for p in PELICULAS) == 5
    assert all(1 <= len(p['generos']) <= 3 for p in PELICULAS)
    assert {'Horror','Romance','Comedy','Music','War','Action','Animation','Science Fiction'} <= set(generos)


def test_sugerencias_no_replican_el_cuestionario():
    validar_catalogo()
    validar_catalogo_k5()
    nombres = {p['titulo'].casefold() for p in PELICULAS}
    assert not nombres.intersection(p['titulo'].casefold() for lista in RECOMENDACIONES.values() for p in lista)
    assert not nombres.intersection(p[0].casefold() for lista in RECOMENDACIONES_K5.values() for p in lista)


def test_menu_sin_radio_y_con_pestanas():
    codigo = (Path(__file__).parent / 'app.py').read_text(encoding='utf-8')
    assert "st.tabs(titulos, key='pestana_activa', on_change='rerun')" in codigo
    assert 'def abrir_seccion(nombre):' in codigo
    assert "st.session_state['pestana_activa'] = nombre" in codigo
    assert "st.radio('Cómo recorrer la aplicación'" not in codigo
