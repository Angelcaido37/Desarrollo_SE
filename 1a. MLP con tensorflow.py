# Importamos las bibliotecas necesarias.
# tensorflow y keras: Las bibliotecas principales.
# models, layers: Módulos para construir el modelo.
# Ejemplo de una red neuronal tipo clasificación
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# scikit-learn: Biblioteca para herramientas de ML como la división de datos.
from sklearn.model_selection import train_test_split
# numpy: Biblioteca para operaciones numéricas.
import numpy as np
# load_iris: Función para cargar el dataset de Iris.
from sklearn.datasets import load_iris

# --- 1. Carga y preparación de los datos ---
# Cargamos el conjunto de datos de flores Iris.
# TensorFlow funciona muy bien con arrays de NumPy, así que los cargamos de esta forma.
iris = load_iris()

# Las características (X) y las etiquetas (y) se cargan directamente.
X = iris.data
y = iris.target

# Dividimos los datos en conjuntos de entrenamiento y prueba.
# `train_test_split` funciona igual que en el ejemplo de PyTorch.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Obtenemos el número de características de entrada y el número de clases.
# Esto hace que el modelo sea dinámico y no dependa de valores fijos.
input_shape = X_train.shape[1]
num_clases = len(np.unique(y))

print(f"Número de características de entrada: {input_shape}")
print(f"Número de clases únicas encontradas en el dataset: {num_clases}")

# --- 2. Definición de la arquitectura del modelo MLP ---
# Usamos `keras.Sequential` para construir un modelo capa por capa.
# Esta es la forma más sencilla de crear una red neuronal secuencial.
model = keras.Sequential([
    # Capa de entrada. No es una capa en sí, sino una forma de definir la forma de la entrada.
    keras.Input(shape=(input_shape,)),
    # Primera capa oculta. Es una capa densa con 10 neuronas.
    # La función de activación `relu` se aplica a la salida de la capa.
    layers.Dense(10, activation='relu'),
    # Segunda capa oculta. Otra capa densa con 10 neuronas y activación `relu`.
    layers.Dense(10, activation='relu'),
    # Capa de salida. Tiene una neurona por cada clase.
    # La activación `softmax` convierte las salidas en probabilidades, sumando 1.
    layers.Dense(num_clases, activation='softmax')
])

# --- 3. Compilación del modelo ---
# `model.compile` es el paso en el que configuramos el proceso de entrenamiento.
model.compile(
    # El optimizador `adam` ajustará los pesos del modelo.
    optimizer='adam',
    # La función de pérdida `SparseCategoricalCrossentropy` se usa para
    # problemas de clasificación con etiquetas que son enteros (0, 1, 2, etc.).
    # A diferencia de PyTorch, la activación `softmax` ya está incluida en la capa de salida.
    loss='sparse_categorical_crossentropy',
    # La métrica `accuracy` medirá la precisión del modelo durante el entrenamiento.
    metrics=['accuracy']
)

# Imprimimos un resumen del modelo para ver su arquitectura y el número de parámetros.
model.summary()

# --- 4. Entrenamiento del modelo ---
# `model.fit` entrena el modelo de forma sencilla y eficiente.
# `epochs`: El número de veces que se pasará todo el conjunto de entrenamiento por el modelo.
# `batch_size`: El número de muestras que se procesan antes de actualizar los pesos.
# `validation_data`: Se utiliza para evaluar el modelo en el conjunto de prueba
#                    después de cada época.
print("\n--- Iniciando el entrenamiento del modelo ---")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1 # Muestra la barra de progreso del entrenamiento.
)
print("--- Entrenamiento finalizado ---")

# --- 5. Evaluación del modelo ---
# `model.evaluate` evalúa la pérdida y la métrica del modelo en el conjunto de prueba.
print("\n--- Evaluando el modelo ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

# --- 6. Ejemplo de predicción para una nueva flor ---
# Creamos un nuevo array de NumPy con las características de una flor.
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# `model.predict` hace la predicción. La salida es un array de probabilidades.
predictions = model.predict(new_flower)

# `np.argmax` obtiene el índice de la clase con la probabilidad más alta.
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Usamos `iris.target_names` para mapear el índice numérico de vuelta al nombre de la flor.
predicted_label = iris.target_names[predicted_class_index]
print(f'\nLa flor con características {new_flower[0]} es un tipo de: {predicted_label}')
