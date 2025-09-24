# Importamos la clase YOLO de la biblioteca ultralytics.
from ultralytics import YOLO
import os
import json
import random
import cv2

# --- 1. Definimos las rutas del dataset ---
# La ruta principal donde se encuentran los datos.
DATA_ROOT = 'datasets/Taco/data/'
# La ruta al archivo de anotaciones JSON.
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, 'annotations.json')

# --- 2. Cargamos el modelo pre-entrenado de YOLOv8 ---
# Usamos el modelo 'yolov8n.pt' que es el más pequeño y rápido.
# Este modelo está pre-entrenado en el dataset COCO y puede detectar
# 80 clases de objetos.
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
    Carga y parsea el archivo de anotaciones del dataset TACO para obtener las rutas de imágenes.
    """
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Obtenemos las rutas de todas las imágenes.
        image_paths = []
        for img in data['images']:
            full_path = os.path.join(DATA_ROOT, img['file_name'])
            # Verificamos que el archivo de imagen realmente exista.
            if os.path.exists(full_path):
                image_paths.append(full_path)
        
        return image_paths
    except FileNotFoundError:
        print(f"Error: El archivo '{annotations_path}' no fue encontrado.")
        print("Asegúrate de que el archivo annotations.json está en la carpeta 'data'.")
        exit()

# Cargamos las rutas de las imágenes.
image_paths = load_taco_data(ANNOTATIONS_PATH)

print(f"Dataset TACO cargado. Se encontraron {len(image_paths)} imágenes.")

# --- 4. Hacemos una predicción de 'clasificación' ---
def classify_image_with_yolo(image_path, model):
    """
    Simula una clasificación de imagen usando un modelo de detección YOLO.
    Elige la clase con la predicción de mayor confianza.
    """
    # Hacemos la predicción con el modelo.
    # El método 'predict' de YOLOv8 devuelve los resultados de la detección.
    results = model(image_path)
    
    # Obtenemos el primer (y único) resultado.
    result = results[0]
    
    # Verificamos si se detectaron objetos.
    if len(result.boxes) == 0:
        return "No se detectaron objetos."

    # Encontramos el objeto con la mayor confianza (score).
    max_score = 0
    best_class_id = -1
    
    # Iteramos a través de todas las predicciones para encontrar la mejor.
    for box in result.boxes:
        score = box.conf[0].item()
        if score > max_score:
            max_score = score
            best_class_id = box.cls[0].item()

    # Mapeamos el ID de la clase al nombre de la clase de COCO.
    best_class_name = model.names[best_class_id]

    # Devolvemos el nombre de la clase y la confianza como un resultado de "clasificación".
    return f"Clase: '{best_class_name}' con {max_score*100:.2f}% de confianza."

# --- Bloque principal de ejecución ---
if __name__ == '__main__':
    # Seleccionamos una imagen aleatoria para la demostración.
    if image_paths:
        random_image_path = random.choice(image_paths)
        print(f"\nProcesando imagen de prueba: {random_image_path}")
        
        # Obtenemos la predicción de "clasificación".
        prediction = classify_image_with_yolo(random_image_path, model)
        
        print("\n--- Resultado de la Clasificación ---")
        print(prediction)
        
        # Opcional: Mostramos la imagen para ver los cuadros de detección.
        img = cv2.imread(random_image_path)
        cv2.imshow("Imagen de prueba", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se encontraron imágenes para procesar. Verifica las rutas en el código.")
