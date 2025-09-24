# Importamos las bibliotecas necesarias.
# tensorflow y keras: Las bibliotecas principales para construir y usar redes neuronales.
import tensorflow as tf
# tensorflow_hub: Para cargar modelos pre-entrenados de alta calidad.
import tensorflow_hub as hub
# numpy: Para operaciones numéricas eficientes.
import numpy as np
# json y os: Para manejar los archivos de datos y las rutas del sistema.
import json
import os
# PIL: Para cargar y manipular las imágenes.
from PIL import Image
# matplotlib: Para visualizar las imágenes y las predicciones.
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. Carga y preparación del dataset TACO ---
# --- Advertencia: Este script asume que has descargado las imágenes del dataset TACO ---
# --- y las has colocado en una carpeta 'taco_dataset' al mismo nivel que este archivo. ---

# Definimos las rutas a los archivos y a las imágenes.
# === CAMBIO AQUÍ: LA CARPETA PRINCIPAL DE IMÁGENES AHORA ES 'data' ===
# El resto de la ruta a la imagen (ej. 'batch_1/000006.jpg') ya está en el archivo JSON.
ANNOTATIONS_PATH = 'datasets/Taco/data/annotations.json'
IMAGES_DIR = 'datasets/Taco/data'
print(f"Buscando anotaciones en: {ANNOTATIONS_PATH}")
print(f"Buscando imágenes en la carpeta: {IMAGES_DIR}")

def load_taco_data(annotations_path):
    """
    Carga y parsea el archivo de anotaciones del dataset TACO.
    Regresa un diccionario con los datos organizados por ID de imagen.
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Creamos un mapeo de ID de imagen a su información.
    images_by_id = {img['id']: img for img in data['images']}
    
    # Creamos un mapeo de ID de categoría a su nombre.
    categories_by_id = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Organizamos las anotaciones por ID de imagen.
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    return images_by_id, categories_by_id, annotations_by_image

# Cargamos los datos.
try:
    images_by_id, categories_by_id, annotations_by_image = load_taco_data(ANNOTATIONS_PATH)
    print("Dataset TACO cargado y parseado exitosamente.")
except FileNotFoundError:
    print("Error: El archivo 'annotations.json' no fue encontrado. Asegúrate de que está en la misma carpeta que el script.")
    exit()

# --- 2. Carga y preprocesamiento de imágenes para la predicción ---
def load_image(image_path, size=(320, 320)):
    """
    Carga una imagen, la redimensiona y la normaliza.
    """
    if not os.path.exists(image_path):
        print(f"Error: La imagen '{image_path}' no fue encontrada.")
        return None
        
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32)
    # Normalizamos los valores de píxeles a un rango de [0, 1].
    img_array = img_array / 255.0
    # Agregamos una dimensión para el batch (el modelo espera un batch de imágenes).
    return np.expand_dims(img_array, axis=0)

# --- 3. Carga del modelo pre-entrenado de TensorFlow Hub ---
# Usamos un modelo SSD (Single Shot MultiBox Detector) MobileNet.
# Este modelo está entrenado en el dataset COCO, lo que lo hace bueno para
# detectar objetos comunes, y podemos adaptarlo para nuestra tarea.
MODEL_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
try:
    detector = hub.load(MODEL_URL)
    print("\nModelo de detección de objetos pre-entrenado cargado exitosamente.")
except Exception as e:
    print(f"\nError al cargar el modelo de TensorFlow Hub: {e}")
    print("Asegúrate de tener conexión a Internet para descargar el modelo.")
    exit()

# --- 4. Ejemplo de predicción y visualización ---
def visualize_results(image_tensor, output, categories):
    """
    Visualiza la imagen con los cuadros delimitadores y las etiquetas de las predicciones.
    """
    # La salida del modelo es un diccionario con las predicciones.
    detection_boxes = output['detection_boxes'][0].numpy()
    detection_classes = output['detection_classes'][0].numpy().astype(int)
    detection_scores = output['detection_scores'][0].numpy()

    # Convertimos la imagen de vuelta a un formato visualizable.
    img_display = (image_tensor[0] * 255).astype(np.uint8)
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_display)
    ax.set_title("Predicción de Detección de Objetos")
    
    # Iteramos sobre las predicciones para dibujar los cuadros y etiquetas.
    # El modelo COCO tiene 90 clases, el dataset TACO tiene muchas más.
    # Aquí solo mostramos las predicciones con un puntaje de confianza alto.
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5: # Umbral de confianza
            ymin, xmin, ymax, xmax = detection_boxes[i]
            score = detection_scores[i]
            class_id = detection_classes[i]
            
            # === CAMBIO AQUÍ: LA ETIQUETA AHORA INCLUYE EL PORCENTAJE DE CONFIANZA ===
            # Obtener el nombre de la clase (este modelo usa etiquetas COCO).
            # Para TACO necesitaríamos remapear las clases, lo cual es un paso avanzado.
            # Por ahora, solo mostraremos el ID y la confianza.
            label = f"Clase ID: {class_id} ({score*100:.2f}%)"
            
            # Convierte las coordenadas normalizadas a píxeles.
            im_height, im_width, _ = img_display.shape
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            
            # Dibuja el cuadro.
            rect = patches.Rectangle((left, top), right - left, bottom - top,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Añade la etiqueta.
            plt.text(left, top - 5, label, color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.show()

# --- Bloque principal de ejecución ---
if __name__ == '__main__':
    # Seleccionamos una imagen aleatoria del dataset para probar.
    img_id_to_test = np.random.choice(list(images_by_id.keys()))
    image_info = images_by_id[img_id_to_test]
    image_file = image_info['file_name']
    
    # === CAMBIO AQUÍ: LA RUTA A LA IMAGEN AHORA INCLUYE LA CARPETA 'data' ===
    image_path = os.path.join(IMAGES_DIR, image_file)

    # Cargamos y preprocesamos la imagen.
    print(f"\nProcesando imagen de prueba: {image_path}")
    test_image_tensor = load_image(image_path)
    if test_image_tensor is not None:
        # Hacemos la predicción con el modelo.
        # Esto solo toma un momento, ya que es una red pre-entrenada.
        print("Haciendo predicción...")
        results = detector(test_image_tensor)
        
        # Visualizamos los resultados.
        print("Visualizando los resultados...")
        visualize_results(test_image_tensor, results, categories_by_id)
        
        # --- Nota didáctica: Entrenamiento ---
        # El entrenamiento de un modelo de detección de objetos es muy complejo y
        # requiere la configuración de una función de pérdida y un bucle de entrenamiento
        # personalizados. No es algo que se pueda mostrar fácilmente en un solo script
        # didáctico. Se necesita un bucle que no solo clasifique, sino que también
        # minimice el error en las coordenadas de las cajas (bounding boxes).
        # Por lo tanto, nos enfocamos en el uso y la visualización de un modelo pre-entrenado.