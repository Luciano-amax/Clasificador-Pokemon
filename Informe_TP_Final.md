# Clasificacion de Pokemon segun su tipo con InceptionV3

Proyecto final de Introduccion a la Inteligencia Artificial - Licenciatura en Ciencias de la Computacion, UNR.

Integrantes: Ignacio Basualdo, Lautaro Capezio y Luciano Duarte.

## 1. Introduccion

El presente trabajo aborda una tarea de clasificacion de imagenes: predecir el tipo primario de un Pokemon a partir de su imagen. Para ello se utilizo el dataset `cnxiaomai/pokemon-classification-gen1-9`, disponible en Hugging Face, y se adapto el modelo InceptionV3 mediante fine-tuning.

La propuesta se vincula con los contenidos vistos en la materia sobre redes neuronales, redes convolucionales profundas y transferencia de aprendizaje. En lugar de entrenar una red desde cero, se parte de un modelo preentrenado sobre ImageNet y se modifican sus ultimas capas para resolver una tarea nueva.

## 2. Conjunto de datos

El dataset contiene imagenes y metadatos de Pokemon de las generaciones 1 a 9. En este trabajo se utiliza la imagen como entrada del modelo y el campo `Type 1` como etiqueta de salida. Esto produce un problema de clasificacion multiclase con 18 clases posibles:

`normal`, `fire`, `water`, `grass`, `electric`, `ice`, `fighting`, `poison`, `ground`, `flying`, `psychic`, `bug`, `rock`, `ghost`, `dragon`, `steel`, `fairy` y `dark`.

La decision de usar solamente `Type 1` simplifica el planteo inicial. Muchos Pokemon poseen dos tipos, pero incorporar `Type 2` cambiaria el problema a clasificacion multi-label, requiriendo otra funcion de perdida, otra codificacion de etiquetas y otras metricas. Por ese motivo se lo deja como una extension futura.

Un aspecto importante es que la distribucion de clases no necesariamente es uniforme. Esto puede sesgar el aprendizaje hacia tipos mas frecuentes y perjudicar tipos con menos ejemplos. Por ese motivo, en el notebook se incluye una grafica de distribucion por clase para entrenamiento y test.

## 3. Metodo utilizado

InceptionV3 es una red convolucional profunda desarrollada para clasificacion de imagenes. El modelo esta compuesto por modulos Inception que combinan convoluciones de distintos tamanos y permiten extraer informacion visual a varias escalas.

La ventaja de utilizar InceptionV3 preentrenado es que las capas iniciales y medias ya aprendieron representaciones visuales generales, como bordes, texturas y formas. Para este trabajo se reemplazo la capa de salida original por una nueva capa lineal con 18 salidas, una por cada tipo Pokemon.

La estrategia de fine-tuning aplicada fue congelar la mayor parte de la red y descongelar solamente el bloque profundo `Mixed_7c`, junto con la capa final principal y la salida auxiliar. Esta eleccion busca un equilibrio: adaptar el modelo al dominio Pokemon sin entrenar demasiados parametros y sin aumentar excesivamente el riesgo de sobreajuste.

## 4. Preprocesamiento

Las imagenes se convierten a RGB porque InceptionV3 espera tres canales de color. Algunas imagenes pueden tener canal alfa, por lo que se aplica una conversion `RGBA -> RGB` para evitar incompatibilidades.

Luego se redimensionan a `299 x 299`, que es el tamano esperado por InceptionV3. Tambien se normalizan con la media y el desvio estandar de ImageNet:

- Media: `[0.485, 0.456, 0.406]`.
- Desvio estandar: `[0.229, 0.224, 0.225]`.

Durante entrenamiento se aplica data augmentation moderado: espejado horizontal, rotacion leve y cambios de brillo/contraste. En validacion no se aplican transformaciones aleatorias, ya que se busca medir sobre datos estables.

## 5. Entrenamiento

La funcion de perdida utilizada fue `CrossEntropyLoss`, apropiada para clasificacion multiclase con una unica etiqueta correcta. El optimizador utilizado fue Adam con learning rates diferenciados:

- `1e-4` para el bloque `Mixed_7c`.
- `1e-3` para la capa final y la salida auxiliar.

Tambien se utilizo un scheduler `StepLR`, que reduce el learning rate a la mitad cada 5 epocas. La corrida registrada tuvo 15 epocas y batch size 64.

## 6. Resultados registrados

La corrida inicial arrojo la siguiente evolucion:

| Epoca | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 3.4197 | 0.2514 | 2.1750 | 0.3368 |
| 2 | 2.8722 | 0.3808 | 1.9728 | 0.3901 |
| 3 | 2.5245 | 0.4740 | 1.8720 | 0.4265 |
| 4 | 2.2560 | 0.5538 | 1.8097 | 0.4478 |
| 5 | 2.0062 | 0.6280 | 1.6223 | 0.5174 |
| 6 | 1.7270 | 0.7113 | 1.5371 | 0.5383 |
| 7 | 1.6011 | 0.7504 | 1.5871 | 0.5383 |
| 8 | 1.5393 | 0.7646 | 1.5481 | 0.5494 |
| 9 | 1.4705 | 0.7865 | 1.5508 | 0.5522 |
| 10 | 1.3839 | 0.8114 | 1.5302 | 0.5597 |
| 11 | 1.2991 | 0.8339 | 1.5211 | 0.5660 |
| 12 | 1.2626 | 0.8486 | 1.5193 | 0.5755 |
| 13 | 1.2493 | 0.8513 | 1.5491 | 0.5779 |
| 14 | 1.2229 | 0.8557 | 1.5495 | 0.5771 |
| 15 | 1.1964 | 0.8699 | 1.5162 | 0.5846 |

El mejor resultado de validacion registrado fue `0.5846` en la epoca 15. Para un problema de 18 clases, este valor supera ampliamente el azar, que seria aproximadamente `1/18 = 0.0556`.

## 7. Discusion

La evolucion de las metricas muestra que el modelo aprende de manera sostenida. La accuracy de entrenamiento crece hasta `0.8699`, mientras que la accuracy de validacion alcanza `0.5846`. Esto indica que la transferencia de aprendizaje funciona como punto de partida, pero tambien aparece una brecha importante entre entrenamiento y validacion.

Dicha brecha sugiere sobreajuste parcial. El modelo logra adaptarse bastante al conjunto de entrenamiento, pero no generaliza con la misma intensidad al conjunto de validacion. Esto puede deberse al tamano del dataset, al desbalance de clases, a la variabilidad estetica entre generaciones y a la ambiguedad propia del problema. Por ejemplo, algunos Pokemon tienen rasgos visuales asociados a su tipo secundario, aunque la etiqueta usada sea solo el tipo primario.

Tambien hay tipos que no tienen una representacion visual directa. Un Pokemon de tipo `normal`, `psychic` o `fairy` puede compartir colores y formas con otros tipos. Esto hace que el problema sea mas dificil que una clasificacion de objetos cotidianos.

## 8. Experimentos recomendados

Para fortalecer el informe final conviene agregar al menos dos corridas comparativas:

| Experimento | Capas entrenables | Hipotesis |
|---|---|---|
| Clasificador solamente | `fc` y `AuxLogits.fc` | Menor sobreajuste, pero posiblemente menor accuracy. |
| Baseline actual | `Mixed_7c`, `fc`, `AuxLogits.fc` | Equilibrio entre adaptacion y generalizacion. |
| Fine-tuning mas profundo | `Mixed_7b`, `Mixed_7c`, `fc`, `AuxLogits.fc` | Puede mejorar si el dominio requiere rasgos mas especificos, aunque aumenta el riesgo de overfitting. |
| Regularizacion adicional | Baseline + weight decay o augmentation mas fuerte | Podria reducir la brecha entre train y validacion. |

Ademas, deberia incorporarse una matriz de confusion y un `classification_report` para analizar precision, recall y f1-score por tipo.

## 9. Metodologia experimental ampliada

El estado actual del proyecto debe interpretarse como un baseline inicial. La primera corrida entreno InceptionV3 con casi toda la red congelada, descongelando solamente el bloque `Mixed_7c` y reemplazando las capas finales `fc` y `AuxLogits.fc`. Esto significa que no se entreno toda la red, sino una parte final del extractor de caracteristicas y las nuevas capas de clasificacion de 18 tipos.

Esta decision es razonable para una primera version porque reduce el costo computacional y evita modificar de manera brusca los pesos preentrenados en ImageNet. Sin embargo, el resultado muestra una brecha considerable entre entrenamiento y validacion: `86.99%` contra `58.46%`. Por lo tanto, el problema principal no parece ser que el modelo no pueda aprender, sino que generaliza de forma limitada.

Para estudiar esto de manera ordenada, el notebook propone las siguientes variantes:

| Experimento | Objetivo | Que deberia mostrar |
|---|---|---|
| `exp_01_classifier_only` | Entrenar solo la cabeza final. | Si el extractor preentrenado alcanza sin adaptar bloques profundos. |
| `exp_02_mixed_7c_dropout_adamw` | Repetir el baseline agregando Dropout y AdamW. | Si la regularizacion reduce la brecha train/validacion. |
| `exp_03_mixed_7b_7c_dropout_adamw` | Descongelar una profundidad mayor. | Si el dominio Pokemon necesita adaptar mas capas visuales. |
| `exp_04_balanced_loss_sampler` | Agregar pesos de clase y sampler balanceado. | Si mejora el desempeno sobre clases minoritarias y el macro-F1. |

### Mecanismos agregados

**Dropout.** Se agrega en la cabeza clasificadora final. Durante entrenamiento apaga aleatoriamente parte de las activaciones, obligando al modelo a no depender de combinaciones demasiado especificas de rasgos. Es una tecnica clasica para reducir sobreajuste.

**AdamW / weight decay.** AdamW aplica una penalizacion sobre los pesos del modelo. Esto ayuda a evitar soluciones con pesos excesivamente grandes y suele mejorar generalizacion.

**Macro-F1.** La accuracy global puede ser enganosa si hay desbalance de clases. Por eso se agrega macro-F1, que promedia el desempeno por clase y da mas informacion sobre tipos minoritarios.

**Pesos por clase y WeightedRandomSampler.** Se agregan como experimento, no como decision fija. Su objetivo es compensar clases con pocos ejemplos. Puede mejorar macro-F1 aunque no necesariamente mejore accuracy global.

**Early stopping.** Se guarda el mejor checkpoint y se corta si la validacion deja de mejorar. Esto evita seguir entrenando cuando solo aumenta la memorizacion del conjunto de entrenamiento.

### Como documentar cada corrida

Cada experimento debe describirse con el mismo formato:

1. Nombre de la corrida.
2. Hipotesis previa.
3. Capas descongeladas.
4. Parametros: dropout, optimizador, learning rates, weight decay, batch size, epocas y scheduler.
5. Mejor resultado de validacion.
6. Brecha entre accuracy de entrenamiento y validacion.
7. Macro-F1 y, si es posible, reporte por clase.
8. Matriz de confusion.
9. Interpretacion: que mejoro, que empeoro y que decision motiva.

Para la defensa oral, es importante remarcar que el accuracy no es el unico criterio. En un problema de 18 clases y etiquetas visualmente ambiguas, un resultado de `58.46%` ya supera ampliamente el azar. La pregunta relevante es que clases aprende, en cuales falla y si las fallas son coherentes con la informacion visual disponible.

## 10. Pendientes estructurales

Aunque el proyecto ya tiene una base funcional, todavia le faltan algunos elementos para quedar como entrega final robusta.

### Funciones que conviene agregar

1. **Evaluacion automatica de un checkpoint.** Una funcion que reciba el nombre de la corrida, cargue el mejor modelo y genere matriz de confusion, `classification_report`, accuracy, macro-F1 y ejemplos de errores. Esto evita evaluar cada corrida de forma manual.

2. **Tabla comparativa final.** Una funcion que lea todos los archivos `*_history.csv` de `outputs/` y arme una tabla unica con mejor epoca, mejor validacion, macro-F1, accuracy final de train, accuracy final de validacion y brecha de generalizacion.

3. **Reporte de errores por clase.** Una funcion que muestre para cada tipo Pokemon cuantos ejemplos tuvo, cuantos acerto, precision, recall y con que otros tipos se confundio mas.

4. **Visualizacion de peores errores.** Una vista con imagen, etiqueta real, etiqueta predicha y nombre del Pokemon. Esto sirve para discutir si el error es absurdo o si visualmente tiene sentido.

5. **Funcion de inferencia individual.** Una funcion `predict_image(...)` que tome una imagen externa o una imagen del dataset y devuelva las probabilidades por tipo. Es util para la presentacion oral.

### Vistas que deberia tener el notebook

El notebook final deberia mostrar, como minimo:

1. Distribucion de clases en train y test.
2. Muestras crudas del dataset.
3. Muestras luego de data augmentation.
4. Curvas de loss y accuracy por corrida.
5. Tabla comparativa de experimentos.
6. Matriz de confusion del mejor modelo.
7. Reporte por clase.
8. Ejemplos cualitativos de aciertos.
9. Ejemplos cualitativos de errores.
10. Si hay tiempo, una vista de probabilidades top-3 para algunos casos.

### Casos a investigar

Para que el informe no quede solamente numerico, conviene investigar casos concretos:

1. **Tipos visualmente claros.** Ver si `fire`, `water`, `grass` o `electric` son reconocidos mejor que tipos menos visuales.

2. **Tipos ambiguos.** Revisar errores en `normal`, `psychic`, `fairy`, `dark`, `dragon` y `flying`, porque muchas veces dependen de convenciones de la franquicia y no solo de la imagen.

3. **Pokemon de doble tipo.** Analizar casos donde el modelo predice algo cercano al `Type 2`, aunque la etiqueta evaluada sea `Type 1`. Esto es muy defendible en la discusion.

4. **Clases minoritarias.** Comparar accuracy y recall de tipos con pocos ejemplos antes y despues del experimento balanceado.

5. **Efecto de descongelar mas capas.** Comparar `Mixed_7c` contra `Mixed_7b_7c`: si mejora validacion, significa que el dominio Pokemon necesitaba adaptar mas representacion; si empeora, probablemente aumento el sobreajuste.

6. **Efecto de regularizacion.** Comparar baseline contra Dropout + AdamW. Aunque no suba mucho el accuracy, es buen resultado si baja la brecha train/validacion o mejora macro-F1.

7. **Top-3 de predicciones.** Puede pasar que el tipo correcto no sea top-1 pero aparezca segundo o tercero. Esto ayuda a mostrar que el modelo no esta completamente perdido.

### Criterio de cierre del proyecto

Para considerar el trabajo listo para entregar, deberiamos tener:

1. Al menos tres corridas documentadas: baseline, regularizada y mas fine-tuning.
2. Una tabla comparativa final.
3. Una matriz de confusion del mejor modelo.
4. Un analisis por clase.
5. Una discusion de errores cualitativos.
6. Una conclusion que no prometa de mas: el modelo aprende, pero la tarea tiene ambiguedad visual y desbalance.

## 11. Conclusiones

Se implemento un clasificador de tipo primario de Pokemon utilizando fine-tuning de InceptionV3. El proyecto cumple con la idea central del enunciado: adaptar un modelo moderno preentrenado a una tarea distinta de la original, experimentar con sus parametros y analizar los resultados obtenidos.

La corrida registrada muestra un desempeno de validacion de `58.46%`, lo cual constituye un baseline razonable para 18 clases. Sin embargo, la diferencia entre entrenamiento y validacion indica que el trabajo todavia puede mejorarse mediante regularizacion, analisis de clases, balanceo y comparacion de configuraciones de fine-tuning.

Como trabajo futuro se propone extender la tarea a clasificacion multi-label usando `Type 1` y `Type 2`, aplicar tecnicas de interpretabilidad como Grad-CAM, comparar con otros modelos preentrenados y estudiar si las confusiones del modelo coinciden con similitudes visuales o con relaciones propias del universo Pokemon.
