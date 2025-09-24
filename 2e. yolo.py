# Importamos la clase YOLO de la biblioteca ultralytics.
from ultralytics import YOLO
import os
import json
import yaml

# --- 1. Definimos las rutas del dataset ---
# La ruta principal donde se encuentran los datos (imágenes y anotaciones).
DATA_ROOT = 'datasets/Taco/data'
# La ruta al archivo de anotaciones COCO JSON.
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, 'annotations.json')

# --- 2. Preparamos el archivo de configuración 'data.yaml' ---
# Este archivo le dirá a YOLO dónde están las imágenes, las anotaciones
# y cuántas clases tiene el dataset.

# Esta función lee las anotaciones para crear el archivo 'data.yaml'.
def create_yaml_file(annotations_path):
    """
    Crea un archivo de configuración YAML necesario para el entrenamiento de YOLOv8.
    """
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        
        # Obtenemos las categorías (clases de basura) y su mapeo.
        names = {cat['id']: cat['name'] for cat in data['categories']}
        
        # El archivo de configuración debe tener este formato.
        # Es lo que el modelo de YOLO espera.
        yaml_content = {
            'path': os.path.abspath(DATA_ROOT),
            'train': 'images', # En este caso, YOLO usará todas las imágenes
            'val': 'images',   # para validación también, ya que no hay una división
            'nc': len(names),
            'names': names
        }
        
        yaml_path = 'taco_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        print(f"Archivo de configuración '{yaml_path}' creado exitosamente.")
        return yaml_path
    except FileNotFoundError:
        print(f"Error: El archivo '{annotations_path}' no fue encontrado.")
        print("Asegúrate de que el archivo annotations.json está en la carpeta 'data'.")
        exit()

# --- 3. Cargamos el modelo pre-entrenado y lo entrenamos con nuestro dataset ---
# Esta es la parte de la transferencia de aprendizaje.
def train_yolo_model(yaml_path):
    """
    Carga un modelo YOLO pre-entrenado y lo entrena con el dataset TACO.
    """
    try:
        print("Cargando el modelo YOLOv8 pre-entrenado para el entrenamiento...")
        # Descargamos el modelo 'yolov8n.pt' para usarlo como base.
        model = YOLO('yolov8n.pt')
        print("Modelo YOLOv8 cargado exitosamente.")

        print("\nIniciando el entrenamiento...")
        # Entrenamos el modelo con los datos que especificamos en el archivo YAML.
        # 'epochs=10' indica que el modelo "verá" el dataset completo 10 veces.
        # 'batch=8' indica que se procesarán 8 imágenes a la vez.
        model.train(data=yaml_path, epochs=10, batch=8, imgsz=640)
        print("¡Entrenamiento finalizado!")
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        print("Asegúrate de tener todas las dependencias instaladas y una GPU si es posible.")

# --- Bloque principal de ejecución ---
if __name__ == '__main__':
    # 1. Creamos el archivo de configuración para YOLO.
    dataset_yaml_path = create_yaml_file(ANNOTATIONS_PATH)
    
    # 2. Entrenamos el modelo.
    train_yolo_model(dataset_yaml_path)

    print("\nEl modelo entrenado se guardó en la carpeta 'runs/detect/train/weights/best.pt'.")
    print("Puedes usar este archivo para hacer predicciones en tus propias imágenes de basura.")
