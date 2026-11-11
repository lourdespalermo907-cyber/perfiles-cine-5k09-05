"""Selección editorial de cinco títulos para cada intersección G-D.

No son predicciones de ratings ni un ranking aprendido de MovieLens. Son ejemplos
curados por el encaje temático y la época de estreno observados en el perfil.
Los 20 títulos son diferentes a los 10 títulos del cuestionario.
"""
from peliculas import PELICULAS

G_AV = 'Animación, familia y aventura'
G_DR = 'Drama, crimen e histórico'
D_MOD = 'Cine de 2000–2010'
D_ANT = 'Cine noventero y clásico'

NOMBRES_CORTOS = {
    (G_AV, D_MOD): 'Animación y aventura · 2000–2010',
    (G_AV, D_ANT): 'Animación y aventura · clásicos y noventas',
    (G_DR, D_MOD): 'Drama, crimen e historia · 2000–2010',
    (G_DR, D_ANT): 'Drama, crimen e historia · clásicos y noventas',
}

RECOMENDACIONES = {
    (G_AV, D_MOD): [
        {'titulo':'Los Increíbles', 'anio':2004, 'generos':'Animación · Acción · Familiar', 'motivo':'Aventura animada y héroes familiares.'},
        {'titulo':'Ratatouille', 'anio':2007, 'generos':'Animación · Comedia · Familiar', 'motivo':'Animación de los 2000 con comedia y personajes cercanos.'},
        {'titulo':'Cómo entrenar a tu dragón', 'anio':2010, 'generos':'Animación · Aventura · Fantasía', 'motivo':'Combina aventura y fantasía en una historia animada.'},
        {'titulo':'Enredados', 'anio':2010, 'generos':'Animación · Aventura · Familiar', 'motivo':'Aventura animada y fantasía de la década de 2010.'},
        {'titulo':'Intensamente', 'anio':2015, 'generos':'Animación · Familiar · Comedia', 'motivo':'Animación y temática familiar, dentro del período del perfil.'},
    ],
    (G_AV, D_ANT): [
        {'titulo':'El rey león', 'anio':1994, 'generos':'Animación · Aventura · Familiar', 'motivo':'Animación familiar y aventura de los noventa.'},
        {'titulo':'Toy Story', 'anio':1995, 'generos':'Animación · Aventura · Familiar', 'motivo':'Historia animada de amistad y aventura noventera.'},
        {'titulo':'Aladdín', 'anio':1992, 'generos':'Animación · Aventura · Fantasía', 'motivo':'Fantasía y aventura animada de los noventa.'},
        {'titulo':'La bella y la bestia', 'anio':1991, 'generos':'Animación · Familiar · Fantasía', 'motivo':'Animación clásica con elementos de fantasía.'},
        {'titulo':'Parque Jurásico', 'anio':1993, 'generos':'Aventura · Ciencia ficción · Suspenso', 'motivo':'Gran aventura clásica con elementos fantásticos.'},
    ],
    (G_DR, D_MOD): [
        {'titulo':'Los infiltrados', 'anio':2006, 'generos':'Crimen · Drama · Suspenso', 'motivo':'Trama criminal con conflicto moral y suspenso.'},
        {'titulo':'Sin lugar para los débiles', 'anio':2007, 'generos':'Crimen · Drama · Suspenso', 'motivo':'Crimen contemporáneo y tensión dramática.'},
        {'titulo':'Zodíaco', 'anio':2007, 'generos':'Crimen · Drama · Misterio', 'motivo':'Investigación criminal y misterio en un relato dramático.'},
        {'titulo':'El discurso del rey', 'anio':2010, 'generos':'Drama · Historia', 'motivo':'Drama histórico ambientado en la monarquía británica.'},
        {'titulo':'12 años de esclavitud', 'anio':2013, 'generos':'Drama · Historia', 'motivo':'Drama de época con trasfondo histórico.'},
    ],
    (G_DR, D_ANT): [
        {'titulo':'Buenos muchachos', 'anio':1990, 'generos':'Crimen · Drama', 'motivo':'Crimen organizado y drama de los noventa.'},
        {'titulo':'Sueños de libertad', 'anio':1994, 'generos':'Drama · Crimen', 'motivo':'Drama carcelario clásico con elementos criminales.'},
        {'titulo':'La lista de Schindler', 'anio':1993, 'generos':'Drama · Historia · Guerra', 'motivo':'Drama histórico ambientado durante la Segunda Guerra Mundial.'},
        {'titulo':'Los Ángeles al desnudo', 'anio':1997, 'generos':'Crimen · Drama · Misterio', 'motivo':'Cine criminal de los noventa con misterio.'},
        {'titulo':'Los intocables', 'anio':1987, 'generos':'Crimen · Drama · Historia', 'motivo':'Relato criminal ambientado en el pasado.'},
    ],
}


def validar_catalogo():
    cuestionario = {p['titulo'].casefold() for p in PELICULAS}
    todos = [p['titulo'].casefold() for lista in RECOMENDACIONES.values() for p in lista]
    assert set(RECOMENDACIONES) == set(NOMBRES_CORTOS), 'Faltan perfiles'
    assert all(len(lista)==5 for lista in RECOMENDACIONES.values()), 'Cada perfil debe contener cinco películas'
    assert len(todos)==len(set(todos))==20, 'Las recomendaciones deben ser 20 títulos distintos'
    assert not set(todos)&cuestionario, 'Una recomendación coincide con el cuestionario'
    assert all(2000 <= f['anio'] <= 2019 for g,d in RECOMENDACIONES if d==D_MOD for f in RECOMENDACIONES[(g,d)])
    assert all(f['anio'] < 2000 for g,d in RECOMENDACIONES if d==D_ANT for f in RECOMENDACIONES[(g,d)])

validar_catalogo()

# El experimento de géneros K=5 no es el modelo principal ni alimenta el cuestionario.
# Este catálogo adicional presenta ejemplos de sus cinco tendencias temáticas.
RECOMENDACIONES_K5 = {
    'Suspenso, misterio, crimen y terror': [
        ('Seven',1995,'Crimen · Misterio · Suspenso'),
        ('El silencio de los inocentes',1991,'Suspenso · Crimen · Terror'),
        ('Los otros',2001,'Terror · Misterio · Suspenso'),
        ('La sospecha',2013,'Crimen · Misterio · Drama'),
        ('¡Huye!',2017,'Terror · Misterio · Suspenso'),
    ],
    'Historia, guerra y drama': [
        ('Dunkerque',2017,'Guerra · Historia · Drama'),
        ('1917',2019,'Guerra · Drama · Historia'),
        ('La vida es bella',1997,'Drama · Guerra'),
        ('La caída',2004,'Historia · Guerra · Drama'),
        ('La patrulla infernal',1957,'Guerra · Drama'),
    ],
    'Acción, ciencia ficción y aventura': [
        ('Matrix',1999,'Acción · Ciencia ficción'),
        ('El origen',2010,'Ciencia ficción · Acción · Suspenso'),
        ('Al filo del mañana',2014,'Acción · Ciencia ficción'),
        ('Mad Max: Furia en el camino',2015,'Acción · Aventura · Ciencia ficción'),
        ('Blade Runner 2049',2017,'Ciencia ficción · Misterio · Drama'),
    ],
    'Animación, familia y fantasía': [
        ('El viaje de Chihiro',2001,'Animación · Fantasía · Aventura'),
        ('WALL·E',2008,'Animación · Familiar · Ciencia ficción'),
        ('Coco',2017,'Animación · Familiar · Fantasía'),
        ('Frozen',2013,'Animación · Familiar · Fantasía'),
        ('Shrek',2001,'Animación · Comedia · Aventura'),
    ],
    'Comedia, romance y perfil mixto': [
        ('Cuando Harry conoció a Sally',1989,'Comedia · Romance'),
        ('10 cosas que odio de ti',1999,'Comedia · Romance'),
        ('La propuesta',2009,'Comedia · Romance'),
        ('Locamente millonarios',2018,'Comedia · Romance'),
        ('Realmente amor',2003,'Comedia · Romance'),
    ],
}


def validar_catalogo_k5():
    titulos=[p[0].casefold() for catalogo in RECOMENDACIONES_K5.values() for p in catalogo]
    preguntas={p['titulo'].casefold() for p in PELICULAS}
    assert len(RECOMENDACIONES_K5)==5
    assert all(len(p)==5 for p in RECOMENDACIONES_K5.values())
    assert len(titulos)==len(set(titulos))==25
    assert not preguntas.intersection(titulos)

validar_catalogo_k5()
