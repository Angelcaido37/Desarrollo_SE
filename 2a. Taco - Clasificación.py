# Importamos las bibliotecas necesarias.
# tensorflow y keras: El corazón de nuestro proyecto de aprendizaje profundo.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
# numpy: Para operaciones numéricas eficientes.
import numpy as np
# os: Para manejar rutas de archivos y carpetas.
import os
# json: Para cargar el archivo de anotaciones.
import json
# PIL: Para cargar y manipular las imágenes.
from PIL import Image
# matplotlib: Para visualizar las imágenes.
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Definición de rutas y carga de datos ---
# La ruta a los archivos de anotación y las imágenes.
ANNOTATIONS_PATH = 'datasets/Taco/data/annotations.json'
IMAGES_DIR = 'datasets/Taco/data'
print(f"Buscando anotaciones en: {ANNOTATIONS_PATH}")
print(f"Buscando imágenes en la carpeta: {IMAGES_DIR}")

def load_taco_data(annotations_path):
    """
    Carga y parsea el archivo de anotaciones del dataset TACO.
    Devuelve un diccionario con los datos organizados.
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Creamos un mapeo de ID de categoría a su nombre.
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    # Obtenemos el número de clases, que será la salida de nuestra red.
    num_classes = len(categories)
    
    # Preparamos las listas para las rutas de las imágenes y sus etiquetas.
    image_paths = []
    labels = []
    
    # Iteramos sobre las imágenes y anotaciones para construir nuestro dataset de entrenamiento.
    # En este script didáctico, solo usamos la primera anotación de cada imagen para simplificar.
    # Un modelo de detección de objetos completo requeriría todas las anotaciones.
    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = [img for img in data['images'] if img['id'] == image_id][0]
        image_path = os.path.join(IMAGES_DIR, image_info['file_name'])
        
        # Obtenemos la etiqueta de la categoría.
        category_id = ann['category_id']
        
        image_paths.append(image_path)
        # La etiqueta debe ser el índice numérico de la categoría.
        labels.append(category_id)
        
    return image_paths, labels, categories, num_classes

# Cargamos los datos del dataset TACO.
try:
    image_paths, labels, categories, num_classes = load_taco_data(ANNOTATIONS_PATH)
    print(f"Dataset TACO cargado y parseado exitosamente. Se encontraron {len(image_paths)} imágenes y {num_classes} categorías.")
except FileNotFoundError:
    print(f"Error: El archivo '{ANNOTATIONS_PATH}' no fue encontrado.")
    print("Asegúrate de que está en la misma carpeta que el script.")
    exit()

# Convertimos las etiquetas a un formato que Keras pueda entender.
labels = np.array(labels)

# --- 2. Preprocesamiento de imágenes para el entrenamiento ---
IMG_SIZE = (224, 224) # El modelo MobileNetV2 espera un tamaño de 224x224.

def preprocess_image(image_path, label):
    """
    Carga, decodifica y preprocesa una imagen para el entrenamiento.
    """
    # Leer la imagen desde la ruta.
    image = tf.io.read_file(image_path)
    # Decodificar la imagen a un tensor.
    image = tf.image.decode_jpeg(image, channels=3)
    # Redimensionar la imagen.
    image = tf.image.resize(image, IMG_SIZE)
    # Convertir a flotantes y normalizar los píxeles.
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Creamos un dataset de TensorFlow para una carga de datos eficiente.
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(preprocess_image).batch(32) # Dividimos los datos en lotes (batches).

# --- 3. Construcción del modelo de Transferencia de Aprendizaje ---
# Usamos un modelo MobileNetV2 pre-entrenado en el dataset ImageNet.
# Le decimos que no incluya la capa de clasificación final, ya que añadiremos la nuestra.
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelamos las capas del modelo base.
# Esto es lo que se conoce como "Transferencia de Aprendizaje". Mantenemos el conocimiento
# del modelo pre-entrenado y solo entrenamos nuestras propias capas.
base_model.trainable = False

# Creamos el modelo de nuestra red.
# Definimos la entrada.
inputs = keras.Input(shape=(224, 224, 3))
# Usamos el modelo base como nuestra primera capa.
x = base_model(inputs, training=False)
# Añadimos una capa para "aplanar" la salida de la CNN.
x = GlobalAveragePooling2D()(x)
# Añadimos una capa densa (fully-connected) que aprenderá a clasificar las 60 clases de basura.
outputs = Dense(num_classes, activation='softmax')(x)
# Creamos el modelo final.
model = Model(inputs, outputs)

# --- 4. Compilación y entrenamiento del modelo ---
# Usamos el optimizador Adam, conocido por su eficiencia.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# La pérdida 'sparse_categorical_crossentropy' es ideal para problemas de clasificación.
# 'accuracy' es nuestra métrica para ver el rendimiento del modelo.
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Imprimimos un resumen de la arquitectura de la red para que la visualices.
model.summary()

print("\n--- Entrenando el modelo ---")
# Entrenamos el modelo con el dataset.
# El número de épocas es el número de veces que el modelo verá todo el dataset.
# Puedes aumentar este número si tienes una GPU.
epochs = 5
history = model.fit(dataset, epochs=epochs)

# --- 5. Predicción y visualización ---
print("\n--- Haciendo una predicción en una imagen de prueba ---")
# Seleccionamos una imagen aleatoria para probar el modelo entrenado.
random_index = np.random.randint(0, len(image_paths))
test_image_path = image_paths[random_index]
true_label = labels[random_index]

# Preparamos la imagen para la predicción.
test_image_tensor = tf.io.read_file(test_image_path)
test_image_tensor = tf.image.decode_jpeg(test_image_tensor, channels=3)
test_image_tensor = tf.image.resize(test_image_tensor, IMG_SIZE)
test_image_tensor = tf.cast(test_image_tensor, tf.float32) / 255.0
test_image_tensor = np.expand_dims(test_image_tensor, axis=0)

# Hacemos la predicción.
predictions = model.predict(test_image_tensor)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Obtenemos los nombres de las categorías para una mejor visualización.
predicted_category_name = categories.get(predicted_class, "Clase desconocida")
true_category_name = categories.get(true_label, "Clase desconocida")

# Visualizamos la imagen con las etiquetas predichas.
plt.imshow(Image.open(test_image_path))
plt.title(f"Predicción: {predicted_category_name} ({confidence*100:.2f}%)\nRealidad: {true_category_name}")
plt.axis('off')
plt.show()

# --- Nota final ---
# Este script se enfoca en la clasificación de imágenes.
# Para hacer una detección de objetos completa, se necesita una arquitectura más compleja
# que regrese tanto la clase como las coordenadas de la caja delimitadora (bounding box).
# El dataset TACO es un excelente punto de partida para futuros estudios en este campo.
