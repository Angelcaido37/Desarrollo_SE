# Importamos las bibliotecas necesarias.
# tensorflow y keras: Las bibliotecas principales para el aprendizaje profundo.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# scikit-learn: Biblioteca para herramientas de ML y generación de datos.
# make_regression: Función para crear un conjunto de datos de regresión.
# train_test_split: Utilizado para dividir los datos en conjuntos de entrenamiento y prueba.
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# numpy: Biblioteca para operaciones numéricas.
import numpy as np

# --- 1. Carga y preparación de los datos ---
# Generamos un conjunto de datos de regresión sintético. Esto simula tener
# datos reales de precios de casas.
# n_samples: El número de casas en el dataset.
# n_features: El número de características de cada casa (ej. tamaño, número de habitaciones).
# noise: Ruido en los datos, para que no sean perfectamente predecibles.
X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=10,
    random_state=42
)
# Escalamos los datos de salida (y) para que simulen precios más realistas.
y = y * 1000 + 50000

print("Datos de características (X) y precios (y) generados exitosamente.")

# Dividimos los datos en conjuntos de entrenamiento y prueba.
# `train_test_split` asegura que el modelo sea evaluado en datos que no ha visto.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Obtenemos el número de características de entrada de forma dinámica.
# Esto es útil para que el modelo sea flexible.
input_shape = X_train.shape[1]
print(f"Número de características de entrada: {input_shape}")

# --- 2. Definición de la arquitectura del modelo MLP ---
# Usamos `keras.Sequential` para construir un modelo capa por capa.
model = keras.Sequential([
    # Capa de entrada. Define la forma de los datos de entrada.
    keras.Input(shape=(input_shape,)),
    # Primera capa oculta densa con 32 neuronas.
    layers.Dense(32, activation='relu'),
    # Segunda capa oculta densa con 16 neuronas.
    layers.Dense(16, activation='relu'),
    # Capa de salida. Esta es la parte clave para la regresión:
    # Tiene solo 1 neurona, ya que solo necesitamos predecir 1 valor (el precio).
    # La activación `linear` (o ninguna) es la predeterminada y es la adecuada
    # para regresión, ya que no limita la salida a un rango específico.
    layers.Dense(1)
])

# --- 3. Compilación del modelo ---
# `model.compile` es el paso en el que configuramos el proceso de entrenamiento.
model.compile(
    # El optimizador `adam` es una opción sólida para la mayoría de los problemas.
    optimizer='adam',
    # La función de pérdida `mean_squared_error` (MSE) es estándar para regresión.
    # Mide el promedio de los errores al cuadrado entre las predicciones y los valores reales.
    loss='mean_squared_error',
    # Las métricas `mae` (error absoluto medio) y `mse` (error cuadrático medio)
    # nos ayudan a entender el rendimiento del modelo durante el entrenamiento.
    metrics=['mae', 'mse']
)

# Imprimimos un resumen del modelo para ver su estructura y el número de parámetros.
model.summary()

# --- 4. Entrenamiento del modelo ---
# `model.fit` entrena el modelo de forma eficiente.
# epochs: Número de pasadas completas por el conjunto de entrenamiento.
# batch_size: Número de muestras procesadas por cada actualización de peso.
# validation_data: Datos de prueba utilizados para evaluar el modelo después de cada época.
print("\n--- Iniciando el entrenamiento del modelo ---")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1  # Muestra el progreso del entrenamiento.
)
print("--- Entrenamiento finalizado ---")

# --- 5. Evaluación del modelo ---
# `model.evaluate` evalúa el rendimiento final del modelo en el conjunto de prueba.
print("\n--- Evaluando el modelo ---")
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida (MSE) en el conjunto de prueba: {loss:.2f}")
print(f"Error Absoluto Medio (MAE) en el conjunto de prueba: {mae:.2f}")

# --- 6. Ejemplo de predicción para una nueva casa ---
# Creamos un nuevo array de NumPy con las características de una casa
# (ej. [característica1, característica2, etc.]).
# Estas características deben estar en el mismo formato que los datos de entrenamiento.
new_house_features = np.array([[-1.0, 0.5, 0.2, -0.8, 1.2]])

# `model.predict` hace la predicción del precio.
predicted_price = model.predict(new_house_features)[0][0]

print(f'\nEl precio predicho para la casa con características {new_house_features[0]} es: ${predicted_price:.2f}')
