# Importamos las bibliotecas necesarias.
from ultralytics import YOLO
import os
import json
import yaml
import shutil
import random
import torch # <-- Importamos torch para la detección de GPU

# --- 1. Definimos las rutas del dataset ---
# La ruta principal donde se encuentran las imágenes y anotaciones del dataset.
DATA_ROOT = 'datasets/Taco/data'
# La ruta al archivo de anotaciones.
ANNOTATIONS_PATH = os.path.join(DATA_ROOT, 'annotations.json')
# La ruta al modelo base pre-entrenado.
MODEL_PATH = 'yolov8n.pt'
# Porcentajes para la división del dataset.
TRAIN_SPLIT = 0.8  # 80% para entrenamiento.
VAL_SPLIT = 0.15   # 15% para validación.
TEST_SPLIT = 0.05  # 5% para prueba (el resto del 100%).

# --- 2. Verificamos la disponibilidad de la GPU ---
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("GPU de NVIDIA encontrada. El entrenamiento y la predicción serán mucho más rápidos.")
else:
    print("No se encontró una GPU. Se usará la CPU.")

# --- 3. Preparamos los directorios de salida ---
def prepare_directories(data_root):
    """Crea la estructura de directorios necesaria para el entrenamiento de YOLO."""
    img_dir = os.path.join(data_root, 'images')
    labels_dir = os.path.join(data_root, 'labels')
    
    # Creamos las subcarpetas para cada conjunto de datos.
    for dir_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(img_dir, dir_name), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, dir_name), exist_ok=True)
        
    print("Estructura de directorios creada exitosamente.")

# --- 4. División y conversión del dataset ---
def split_dataset_and_convert_annotations(annotations_path, data_root):
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba,
    y convierte las anotaciones al formato de YOLO.
    """
    print("\nIniciando la división del dataset y la conversión de anotaciones...")
    
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    images_dict = {img['id']: img for img in coco_data['images']}
    annotations_dict = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_dict:
            annotations_dict[img_id] = []
        annotations_dict[img_id].append(ann)

    # Obtenemos los nombres y el número de clases.
    class_names = [cat['name'] for cat in coco_data['categories']]
    num_classes = len(class_names)
    
    # Obtenemos los IDs de las imágenes que tienen anotaciones.
    image_ids_with_annotations = list(annotations_dict.keys())
    random.shuffle(image_ids_with_annotations)
    
    num_images = len(image_ids_with_annotations)
    train_split_index = int(num_images * TRAIN_SPLIT)
    val_split_index = train_split_index + int(num_images * VAL_SPLIT)
    
    train_ids = image_ids_with_annotations[:train_split_index]
    val_ids = image_ids_with_annotations[train_split_index:val_split_index]
    test_ids = image_ids_with_annotations[val_split_index:]
    
    for dataset_type, image_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        print(f"Procesando {len(image_ids)} imágenes para el conjunto de {dataset_type}...")
        for img_id in image_ids:
            img_info = images_dict[img_id]
            file_name = img_info['file_name']
            img_path_src = os.path.join(data_root, file_name)
            
            # Copiamos la imagen al directorio correspondiente.
            img_path_dest = os.path.join(data_root, 'images', dataset_type, os.path.basename(file_name))
            
            # Creamos el archivo de anotaciones de YOLO.
            labels_path_dest = os.path.join(data_root, 'labels', dataset_type, os.path.splitext(os.path.basename(file_name))[0] + '.txt')
            
            if os.path.exists(img_path_src):
                shutil.copyfile(img_path_src, img_path_dest)
            
            with open(labels_path_dest, 'w') as f:
                if img_id in annotations_dict:
                    for ann in annotations_dict[img_id]:
                        x, y, w, h = ann['bbox']
                        img_width = img_info['width']
                        img_height = img_info['height']
                        
                        # Convertimos las coordenadas al formato YOLO.
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        norm_width = w / img_width
                        norm_height = h / img_height
                        
                        # El ID de la clase debe comenzar en 0.
                        class_id = ann['category_id'] - 1
                        
                        f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

    print("División del dataset y conversión de anotaciones completada.")
    return class_names, num_classes

# --- 5. Preparamos el archivo de configuración para el entrenamiento ---
def create_yaml_config(data_root, class_names, num_classes):
    """Crea el archivo de configuración .yaml para el entrenamiento de YOLO."""
    data_config = {
        'path': os.path.join(os.getcwd(), data_root),
        'train': 'images/train',
        'val': 'images/val',
        'names': class_names,
        'nc': num_classes,
    }

    config_path = 'taco_data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"Archivo de configuración '{config_path}' creado exitosamente.")
    return config_path

# --- 6. Ejecutamos el entrenamiento ---
def train_yolo_model(config_file, model_path, epochs, imgsz, device):
    """Carga y entrena el modelo YOLO."""
    try:
        print("\nCargando el modelo base YOLOv8 para el entrenamiento...")
        model = YOLO(model_path)
        print("Modelo base YOLOv8 cargado exitosamente.")

        print("\nIniciando el entrenamiento del modelo...")
        results = model.train(data=config_file, epochs=epochs, imgsz=imgsz, device=device)

        print("\nEntrenamiento finalizado. El modelo entrenado se encuentra en 'runs/detect/train/weights/best.pt'.")
        return results

    except Exception as e:
        print(f"\nOcurrió un error durante el entrenamiento: {e}")
        print("Por favor, asegúrate de que todos los archivos y carpetas existen y son accesibles.")
        
# --- 7. Proceso principal ---
if __name__ == '__main__':
    # Llamamos a las funciones para preparar el dataset.
    prepare_directories(DATA_ROOT)
    class_names, num_classes = split_dataset_and_convert_annotations(ANNOTATIONS_PATH, DATA_ROOT)
    
    # Creamos y obtenemos la ruta del archivo de configuración.
    config_path = create_yaml_config(DATA_ROOT, class_names, num_classes)
    
    # Entrenamos el modelo.
    train_yolo_model(config_path, MODEL_PATH, epochs=120, imgsz=640, device=device)
