# Importamos la clase YOLO de la biblioteca ultralytics.
from ultralytics import YOLO
import os
import json
import random
import cv2
import matplotlib.pyplot as plt

# --- 1. Definimos las rutas del dataset ---
# La ruta principal donde se encuentran los datos.
DATA_ROOT = 'datasets/Taco/data/'
# La ruta al archivo de anotaciones JSON.
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, 'annotations.json')

# --- 2. Cargamos el modelo pre-entrenado de YOLOv8 ---
# Usamos el modelo 'yolov8n.pt' que es el más pequeño y rápido.
# Este modelo ya está pre-entrenado en el dataset COCO y puede detectar
# 80 clases de objetos. Lo usaremos como base.
try:
    print("Cargando el modelo YOLOv8 pre-entrenado...")
    model = YOLO('yolov8n.pt')
    print("Modelo YOLOv8 cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de tener conexión a Internet y de que el modelo 'yolov8n.pt' pueda ser descargado.")
    exit()

# --- 3. Cargamos los datos del dataset TACO ---
def load_taco_data(annotations_path):
    """
    Carga y parsea el archivo de anotaciones del dataset TACO para obtener las categorías y rutas de imágenes.
    """
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Creamos un diccionario para mapear el ID de la categoría a su nombre.
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Obtenemos las rutas de todas las imágenes.
        image_paths = []
        for img in data['images']:
            full_path = os.path.join(DATA_ROOT, img['file_name'])
            if os.path.exists(full_path):
                image_paths.append(full_path)
        
        return image_paths, categories
    except FileNotFoundError:
        print(f"Error: El archivo '{annotations_path}' no fue encontrado.")
        print("Asegúrate de que el archivo annotations.json está en la carpeta 'data'.")
        exit()

# Cargamos las rutas y categorías.
image_paths, categories = load_taco_data(ANNOTATIONS_PATH)

print(f"Dataset TACO cargado. Se encontraron {len(image_paths)} imágenes.")

# --- 4. Hacemos una predicción y visualizamos el resultado ---
def visualize_predictions(image_path, model, categories):
    """
    Realiza una predicción en una imagen y visualiza los resultados con los cuadros delimitadores.
    """
    # Hacemos la predicción con el modelo.
    # El método 'predict' de YOLOv8 es muy eficiente.
    results = model(image_path)
    
    # Obtenemos el primer (y único) resultado, ya que solo procesamos una imagen.
    result = results[0]
    
    # Obtenemos la imagen original usando OpenCV para dibujar sobre ella.
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convertimos el color para Matplotlib.

    # Dibuja los cuadros y las etiquetas.
    for box in result.boxes:
        # Extraemos las coordenadas, la confianza y la clase.
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        score = round(box.conf[0].item(), 2)
        class_id = box.cls[0].item()
        
        # Como nuestro modelo YOLO no ha sido entrenado en TACO, las clases
        # que predice son las del dataset COCO (80 clases).
        # Para ver un nombre de clase, necesitamos usar el mapeo de COCO.
        # Ultralytics ya incluye estos nombres en su objeto de resultados.
        predicted_class_name = model.names[class_id]
        
        # Definimos el color del cuadro.
        color = (255, 0, 0)
        # Dibujamos el cuadro delimitador en la imagen.
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Preparamos la etiqueta con el nombre de la clase y la confianza.
        label = f"{predicted_class_name} ({score:.2f})"
        
        # Dibujamos el texto de la etiqueta en la imagen.
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostramos la imagen usando Matplotlib.
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.title("Predicciones de YOLOv8")
    plt.axis('off')
    plt.show()

# --- Bloque principal de ejecución ---
if __name__ == '__main__':
    # Seleccionamos una imagen aleatoria para la demostración.
    if image_paths:
        random_image_path = random.choice(image_paths)
        print(f"\nProcesando imagen de prueba: {random_image_path}")
        visualize_predictions(random_image_path, model, categories)
    else:
        print("No se encontraron imágenes para procesar. Verifica las rutas en el código.")
