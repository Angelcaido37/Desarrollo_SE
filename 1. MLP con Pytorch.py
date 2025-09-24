# Importamos la clase YOLO de la biblioteca ultralytics.
from ultralytics import YOLO
import os
import random
import cv2
import matplotlib.pyplot as plt

# --- 1. Definimos las rutas ---
# La ruta a nuestro modelo entrenado, guardado en la carpeta 'runs'.
# La ruta puede variar, revisa la salida de tu entrenamiento.
# Generalmente se encuentra en 'runs/detect/train/weights/best.pt'.
MODEL_PATH = 'runs/detect/train/weights/best.pt'
# La ruta principal donde se encuentran las imágenes del dataset TACO.
DATA_ROOT = 'data'

# --- 2. Cargamos nuestro modelo entrenado ---
# Usamos el modelo 'best.pt' que es el resultado de nuestro entrenamiento.
try:
    print("Cargando el modelo entrenado...")
    model = YOLO(MODEL_PATH)
    print("Modelo entrenado cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que el archivo 'best.pt' existe en la ruta especificada.")
    exit()

# --- 3. Obtenemos una imagen aleatoria para la predicción ---
def get_random_image_path(root_dir):
    """
    Busca una imagen aleatoria en la estructura de carpetas 'data/batch_N'.
    """
    # Buscamos en todas las subcarpetas dentro de DATA_ROOT.
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(dirpath, filename))
    
    if not image_paths:
        return None
        
    return random.choice(image_paths)

# Obtenemos la ruta a una imagen de prueba.
random_image_path = get_random_image_path(DATA_ROOT)

if not random_image_path:
    print("No se encontraron imágenes en la carpeta 'data'. Asegúrate de que los archivos están ahí.")
    exit()

print(f"\nProcesando imagen de prueba: {random_image_path}")

# --- 4. Hacemos la predicción y la visualizamos ---
def visualize_predictions(image_path, model):
    """
    Realiza una predicción en una imagen usando el modelo entrenado y muestra el resultado.
    """
    # Hacemos la predicción. El modelo dibuja los cuadros y etiquetas automáticamente.
    results = model(image_path)
    
    # El método plot() de Ultralytics ya nos da la imagen con los cuadros dibujados.
    result_img = results[0].plot()

    # Convertimos la imagen de BGR a RGB para Matplotlib.
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Mostramos la imagen usando Matplotlib.
    plt.figure(figsize=(12, 12))
    plt.imshow(result_img_rgb)
    plt.title("Predicción con el modelo entrenado")
    plt.axis('off')
    plt.show()

# Ejecutamos la visualización de la predicción.
visualize_predictions(random_image_path, model)

print("\nPredicción finalizada. Se mostró la imagen con los cuadros de basura detectada.")
