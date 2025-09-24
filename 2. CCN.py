# Importamos las bibliotecas necesarias.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Carga y preparación del dataset Fashion MNIST ---
# Este dataset ya está incluido en Keras, lo que facilita su carga.
# Contiene 60,000 imágenes para entrenamiento y 10,000 para prueba.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Los nombres de las 10 clases que corresponden a las etiquetas numéricas.
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Imprimimos las dimensiones de los datos para entender su forma.
print(f"Dimensiones de las imágenes de entrenamiento: {train_images.shape}")
print(f"Dimensiones de las etiquetas de entrenamiento: {train_labels.shape}")
print(f"Número de clases: {len(class_names)}")

# --- 2. Preprocesamiento de los datos ---
# Normalizamos los valores de los píxeles de 0-255 a 0-1.
# Esto es crucial para que la red funcione correctamente, ya que facilita el entrenamiento.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Añadimos una dimensión al final para el "canal de color".
# Las CNNs esperan un formato (altura, anchura, canales).
# Las imágenes en blanco y negro tienen 1 canal, las de color tienen 3 (RGB).
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# --- 3. Definición de la arquitectura del modelo CNN ---
# Usamos `keras.Sequential` para construir la red capa por capa.
model = keras.Sequential([
    # Capa convolucional. Detecta características como bordes y texturas.
    # 32: Número de filtros (detectores de características).
    # (3, 3): Tamaño de cada filtro.
    # activation='relu': Activa las neuronas si el resultado es positivo.
    # input_shape: La forma de las imágenes de entrada (28x28 píxeles con 1 canal de color).
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Capa de max pooling. Reduce las dimensiones de la imagen, manteniendo solo las características más importantes.
    layers.MaxPooling2D((2, 2)),
    
    # Segunda capa convolucional para detectar características más complejas.
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Segunda capa de max pooling.
    layers.MaxPooling2D((2, 2)),
    
    # Tercera capa convolucional.
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # `Flatten`: Convierte la matriz de características en un vector unidimensional.
    # Esto es necesario para conectar las capas convolucionales con las capas densas (MLP).
    layers.Flatten(),
    
    # Capa densa (MLP) para la clasificación.
    layers.Dense(64, activation='relu'),
    
    # Capa de salida. 10 neuronas para las 10 clases de ropa.
    # `softmax`: Convierte las salidas en probabilidades.
    layers.Dense(10, activation='softmax')
])

# Imprimimos un resumen del modelo para ver la arquitectura.
model.summary()

# --- 4. Compilación del modelo ---
# `adam`: Optimizador para el entrenamiento.
# `sparse_categorical_crossentropy`: Función de pérdida para clasificación de múltiples clases.
# `accuracy`: Métrica para evaluar la precisión del modelo.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 5. Entrenamiento del modelo ---
print("\n--- Iniciando el entrenamiento del modelo ---")
model.fit(train_images, train_labels, epochs=5, verbose=1)
print("--- Entrenamiento finalizado ---")

# --- 6. Evaluación del modelo ---
print("\n--- Evaluando el modelo ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nPrecisión en el conjunto de prueba: {test_acc:.4f}")

# --- 7. Ejemplo de predicción ---
# Seleccionamos una imagen de prueba al azar.
img_index = np.random.randint(0, len(test_images))
img = test_images[img_index]
true_label = test_labels[img_index]

# `expand_dims` agrega una dimensión para que coincida con el formato de entrada esperado por el modelo.
# El modelo espera un batch de imágenes (aunque sea de una sola).
img_to_predict = np.expand_dims(img, axis=0)

# Hacemos la predicción.
predictions = model.predict(img_to_predict)

# Obtenemos la clase predicha (la que tiene la probabilidad más alta).
predicted_label = np.argmax(predictions[0])

# Imprimimos los resultados y mostramos la imagen.
print(f"\nEtiqueta real: {true_label} ({class_names[true_label]})")
print(f"Predicción del modelo: {predicted_label} ({class_names[predicted_label]})")
print(f"Probabilidades de la predicción: {predictions[0]}")

# Código para mostrar la imagen (requiere matplotlib).
plt.figure()
plt.imshow(img.squeeze(), cmap=plt.cm.binary)
plt.title(f"Etiqueta real: {class_names[true_label]}, Predicción: {class_names[predicted_label]}")
plt.show()
