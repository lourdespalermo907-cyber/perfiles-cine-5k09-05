"""Películas del cuestionario: variedad de géneros y 5 títulos anteriores a 2000 / 5 posteriores.

Los metadatos son una selección simplificada para la demostración: una película
puede tener más géneros que los incluidos en este catálogo. Nunca se puntúa
«No la vi» como cero y las recomendaciones no repiten estos títulos.
"""
PELICULAS = [
    {'id': 'godfather', 'titulo': 'El padrino', 'anio': 1972,
     'generos': ['Crime', 'Drama']},
    {'id': 'shining', 'titulo': 'El resplandor', 'anio': 1980,
     'generos': ['Horror', 'Thriller']},
    {'id': 'back_to_future', 'titulo': 'Volver al futuro', 'anio': 1985,
     'generos': ['Adventure', 'Comedy', 'Science Fiction']},
    {'id': 'titanic', 'titulo': 'Titanic', 'anio': 1997,
     'generos': ['Drama', 'Romance']},
    {'id': 'private_ryan', 'titulo': 'Rescatando al soldado Ryan', 'anio': 1998,
     'generos': ['Drama', 'History', 'War']},
    {'id': 'school_of_rock', 'titulo': 'Escuela de rock', 'anio': 2003,
     'generos': ['Comedy', 'Music']},
    {'id': 'finding_nemo', 'titulo': 'Buscando a Nemo', 'anio': 2003,
     'generos': ['Animation', 'Family']},
    {'id': 'john_wick', 'titulo': 'John Wick', 'anio': 2014,
     'generos': ['Action', 'Crime', 'Thriller']},
    {'id': 'arrival', 'titulo': 'La llegada', 'anio': 2016,
     'generos': ['Drama', 'Science Fiction', 'Mystery']},
    {'id': 'glass_onion', 'titulo': 'Glass Onion: Un misterio de Knives Out', 'anio': 2022,
     'generos': ['Mystery', 'Comedy', 'Crime']},
]
