# Importamos las bibliotecas necesarias.
# tensorflow y keras: Las bibliotecas principales para construir y entrenar la red.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# scikit-learn: Biblioteca para herramientas de ML y generación de datos.
# make_classification: Función para crear un conjunto de datos de clasificación sintético.
# train_test_split: Utilizado para dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# numpy: Biblioteca para operaciones numéricas.
import numpy as np

# --- 1. Generación y preparación de los datos ---
# Generamos un conjunto de datos de clasificación sintético para simular
# las estadísticas de partidos de fútbol.
# n_samples: El número total de partidos.
# n_features: El número de características por partido (ej. puntos, racha de victorias, etc.).
# n_classes: El número de resultados posibles (victoria, empate, derrota).
# random_state: Asegura que los datos generados sean los mismos cada vez que se ejecute el script.
X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_classes=3,
    n_informative=3, # Cuántas de las características son útiles para la clasificación.
    n_redundant=0, # Cuántas características son redundantes.
    random_state=42
)

# Convertimos las etiquetas numéricas a nombres más descriptivos.
etiquetas = {0: "Victoria del equipo local", 1: "Empate", 2: "Derrota del equipo local"}
print(f"Datos generados. El objetivo es clasificar el partido en una de estas {len(etiquetas)} categorías.")

# Dividimos los datos en conjuntos de entrenamiento y prueba.
# `train_test_split` separa los datos para que el modelo aprenda con un 80%
# y sea evaluado con el 20% restante.
= train_test_split(X, y, test_size=0.2, random_state=42)

# === Sección para inspeccionar los datos generados (NUEVAS LÍNEAS) ===
# `X_train` contiene los datos de entrenamiento (características).
print("\n--- Vista previa de las primeras 5 filas de las características de entrenamiento (X_train) ---")
print(X_train[:5]) # Muestra las primeras 5 filas.

# `y_train` contiene las etiquetas de entrenamiento (los resultados de los partidos).
print("\n--- Vista previa de las primeras 5 etiquetas de entrenamiento (y_train) ---")
print(y_train[:5]) # Muestra las primeras 5 etiquetas.

# Obtenemos el número de características de entrada y el número de clases.
input_shape = X_train.shape[1]
num_clases = len(np.unique(y))
print(f"\nNúmero de características de entrada: {input_shape}")
print(f"Número de clases de salida: {num_clases}")

# --- 2. Definición de la arquitectura del modelo MLP ---
# Usamos `keras.Sequential` para construir el modelo capa por capa.
model = keras.Sequential([
    # Capa de entrada. Define la forma de los datos de entrada.
    keras.Input(shape=(input_shape,)),
    # Primera capa oculta densa con 64 neuronas.
    layers.Dense(64, activation='relu'),
    # Segunda capa oculta densa con 32 neuronas.
    layers.Dense(32, activation='relu'),
    # Capa de salida. Esta es la parte clave para la clasificación:
    # Tiene un número de neuronas igual al número de clases.
    # La activación `softmax` convierte las salidas en probabilidades que suman 1.
    layers.Dense(num_clases, activation='softmax')
])

# --- 3. Compilación del modelo ---
# `model.compile` configura el proceso de entrenamiento.
model.compile(
    # Optimizador `adam` para ajustar los pesos del modelo.
    optimizer='adam',
    # Función de pérdida `sparse_categorical_crossentropy` para problemas de clasificación.
    loss='sparse_categorical_crossentropy',
    # La métrica `accuracy` para medir la precisión del modelo.
    metrics=['accuracy']
)

# Imprimimos un resumen del modelo para ver su estructura.
model.summary()

# --- 4. Entrenamiento del modelo ---
# `model.fit` entrena el modelo.
# epochs: Número de pasadas completas por el conjunto de entrenamiento.
# validation_data: Datos de prueba para evaluar el modelo después de cada época.
print("\n--- Iniciando el entrenamiento del modelo ---")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1 # Muestra el progreso del entrenamiento.
)
print("--- Entrenamiento finalizado ---")

# --- 5. Evaluación del modelo ---
# `model.evaluate` evalúa el rendimiento final en el conjunto de prueba.
print("\n--- Evaluando el modelo ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

# --- 6. Ejemplo de predicción para un nuevo partido ---
# Creamos un nuevo array de NumPy con las características de un partido
# que el modelo nunca ha visto.
nuevo_partido = np.array([[-1.2, 0.5, 0.8, -0.3, 1.0]])

# `model.predict` hace la predicción. La salida son las probabilidades para cada clase.
prediccion_probabilidades = model.predict(nuevo_partido)

# `np.argmax` encuentra el índice de la clase con la probabilidad más alta.
clase_predicha_indice = np.argmax(prediccion_probabilidades, axis=1)[0]

# Mapeamos el índice numérico de vuelta al resultado del partido.
resultado_predicho = etiquetas[clase_predicha_indice]
print(f'\nLas probabilidades para el partido son: {prediccion_probabilidades[0]}')
print(f'El resultado predicho es: {resultado_predicho}')
