# Perfiles cinematográficos — UTN · Grupo 5K09-05

## Pregunta del proyecto

¿Es posible identificar perfiles de usuarios a partir de sus patrones de valoración de películas y utilizar dichos perfiles para orientar recomendaciones de películas?

**Resultado demostrado:** se identificaron **cuatro perfiles descriptivos** mediante dos PCA y dos K-means independientes (dos grupos de géneros y dos de décadas). **No está demostrada la calidad predictiva de las recomendaciones.**

## Secciones de la aplicación

- **El proyecto:** portada con tres números; fuente y filtro de usuarios; hallazgos de géneros y décadas; comparación de modelos y «Cómo lo sabemos» en cada sección.
- **Explorá:** todos los usuarios en scatter interactivo Altair; filtros por perfil, bloque PCA y cantidad de valoraciones; detalle del usuario seleccionado y su cluster.
- **Probá el modelo:** diez películas con notas de 1 a 5 o «No la vi», conversión a la estructura original, asignación **explícita** de perfil y cinco recomendaciones nuevas del perfil.
- **Películas por perfil:** cinco películas para cada una de las cuatro intersecciones principales y, aparte, un listado ilustrativo de cinco películas para cada una de las cinco tendencias del modelo exploratorio K=5. Todos los títulos difieren de los del cuestionario.

Las **20 sugerencias principales** y las **25 exploratorias** son seleccionadas editorialmente por época y género; no se obtuvieron con una evaluación de un recomendador ni se calculó precisión@5.

## Navegación

Las **pestañas superiores** y los **enlaces de texto del menú lateral** abren las mismas cuatro secciones. Ambas formas de navegación están sincronizadas, sin botones de radio. La selección superior también actualiza el indicador del enlace lateral.

El cuestionario distribuye sus diez títulos entre varios géneros. Solo **una película** incluye Aventura en este catálogo, cinco fueron estrenadas **antes de 2000** y cinco **desde 2000**. Las películas del cuestionario no se repiten entre las sugerencias de los perfiles.

En «Lo que encontramos» la matriz 2 × 2 usa nombres abreviados para evitar superposiciones, muestra cantidades y porcentajes y conserva los nombres completos en la información al pasar el cursor.


## Archivos

| Archivo | Uso |
|---|---|
| `app.py` | aplicación Streamlit, storytelling y visualización Altair |
| `modelado.py` | preprocesamiento, PCA y K-means reproducibles, inferencia de los casos |
| `peliculas.py` | diez títulos del cuestionario con año y géneros |
| `recomendaciones.py` | cinco sugerencias por perfil, más las cinco tendencias exploratorias |
| `usuarios_generos.csv` | dataset Silver de 5.000 usuarios proveniente del pipeline |
| `build_app.py` | regenera etiquetas, resumen y métodos a partir del CSV actualizado |
| `test_entrega4.py` | pruebas automáticas de consistencia |
| `requirements.txt` | dependencias Python |
| `resultados_reproducibles/` | salidas creadas por `build_app.py` |

## Modelado exacto usado

- Usuarios con `cantidad_valoraciones >= 150` → **3.503** de 5.000.
- Géneros: puntuaciones relativas a la media general, suavizado `n/(n+15)`, cobertura mínima del 75 %, `StandardScaler`, PCA para conservar ≥80 % de varianza (8 PCs), K-means `k=2`.
- Décadas: proporciones de películas calificadas por década de estreno, transformación `arcsin(sqrt(p))`, 2020+ excluida del espacio de distancia por baja cobertura, `StandardScaler`, PCA (3 PCs), K-means `k=2`.
- Ambas dimensiones: `random_state=42`, `n_init=10`, **cuatro intersecciones** G-D. K=5 por géneros permanece exploratorio. El modelo conjunto se conserva como comparación narrativa, no para inferir.
- El modelado es reproducible con las cifras del notebook: 1.096, 1.108, 647 y 652 usuarios por perfil.

## Limitaciones del cuestionario

Se entrenó con al menos 150 películas por usuario, pero el cuestionario pide entre 6 y 10. Las inferencias son **provisionales**. Si dos personas valoran las mismas películas, las **proporciones por década son iguales aunque sus estrellas difieran**. No hay probabilidades calibradas y la muestra fija no permite validar preferencias personales.


## Reproducibilidad e integración

Para regenerar desde un CSV actualizado del pipeline, reemplazar `usuarios_generos.csv`, ejecutar `python build_app.py` y reiniciar la app. El módulo de modelos se vuelve a ajustar con la misma semilla y transforma los nuevos datos al iniciar. La selección editorial de películas se mantiene fija a menos que se revise manualmente.

En el repositorio integrado de la cátedra también deben estar los DAG de Airflow, la EDA, el notebook de la Entrega 3 y un README de nivel raíz que explique el recorrido completo de regeneración.

**Cuidado conceptual:** las variables temporales del modelo indican épocas de **estreno de títulos calificados**, no fechas en que un usuario puntuó ni evidencia de que disfrute más una década. Los perfiles temáticos están definidos por **afinidades relativas** entre grupos; son descripciones, no etiquetas verdaderas.
