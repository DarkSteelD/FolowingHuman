
from flask import Flask, request
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import threading

app = Flask(__name__)
model = YOLO("yolov9c.yaml")

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9c.pt")

# Display model information (optional)
model.info() # Инициализируйте YOLO с нужной моделью (замените на нужную)

# Глобальные переменные для последнего изображения и боксов
last_image = None
last_boxes = None

# Функция для обработки детекции и рисования боксов
def detect_and_draw(image):
    global last_image, last_boxes
    cv_image = cv2.resize(image, (1080, 1080))
    
    # Выполняем детекцию объектов
    results = model(cv_image)
    boxes = []
    
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, box[:4])
        width = x_max - x_min
        height = y_max - y_min
        boxes.append({"x": x_min, "y": y_min, "width": width, "height": height})
    
    # Сохраняем последнее изображение и боксы
    last_image = cv_image
    last_boxes = boxes

    # Рисуем боксы на изображении
    for box in boxes:
        x, y, width, height = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(cv_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return cv_image

# Эндпоинт для загрузки изображения
@app.route('/upload', methods=['POST'])
def upload_image():
    global last_image, last_boxes
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    detect_and_draw(image)
    return {"status": "Image received and processed"}

# Функция для Gradio интерфейса, которая показывает последнее изображение
def show_last_image():
    if last_image is None:
        return None, {"detected_boxes": []}
    return last_image, {"detected_boxes": last_boxes}

# Интерфейс Gradio для отображения последнего изображения и JSON с боксами
interface = gr.Interface(
    fn=show_last_image,
    inputs=None,
    outputs=[gr.Image(type="numpy"), "json"],
    description="Displays the last uploaded image with detection boxes."
)

# Запуск Flask и Gradio параллельно
def run_flask():
    app.run(host="0.0.0.0", port=5000)

def run_gradio():
    interface.launch(server_name="0.0.0.0", server_port=7860)

# Запуск Flask и Gradio в отдельных потоках
threading.Thread(target=run_flask).start()
threading.Thread(target=run_gradio).start()
