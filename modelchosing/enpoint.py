from flask import Flask, request, jsonify
import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Инициализируем модель YOLO
model = YOLO("yolov9c.pt")
model.info()

# Глобальные переменные для последнего изображения и боксов
last_image = None
last_boxes = None

# Функция для обработки детекции и вычисления направления движения
def detect_and_draw(image):
    global last_image, last_boxes
    cv_image = cv2.resize(image, (1080, 1080))

    # Выполняем детекцию объектов
    results = model(cv_image)
    boxes = []
    target_class = 'cup'
    target_box = None

    # Проходим по всем обнаруженным объектам
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        width = x_max - x_min
        height = y_max - y_min
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Если класс - 'person', сохраняем его для вычисления направления
        if class_name == target_class:
            target_box = (x_min, y_min, x_max, y_max)
            # Рисуем рамку для 'person'
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(cv_image, class_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            break  # Останавливаемся на первом найденном 'person'

        # Сохраняем координаты боксов всех объектов
        boxes.append({"x": x_min, "y": y_min, "width": width, "height": height, "class": class_name})
        print({"x": x_min, "y": y_min, "width": width, "height": height, "class": class_name})
    direction = 1  # По умолчанию - назад

    if target_box:
        x_min, y_min, x_max, y_max = target_box
        box_center_x = (x_min + x_max) // 2
        box_center_y = (y_min + y_max) // 2
        frame_center_x, frame_center_y = cv_image.shape[1] // 2, cv_image.shape[0] // 2

        # Определяем направление движения на основе положения 'person'
        if abs(box_center_x - frame_center_x) < cv_image.shape[1] * 0.1:
            # Если 'person' в центре по горизонтали
            direction = 0  # Прямо
        elif box_center_x < frame_center_x - frame_center_x // 2:
            direction = 3  # Влево
        elif box_center_x > frame_center_x + frame_center_x // 2:
            direction = 2  # Вправо
        else:
            direction = 1 # назад
        # Вы можете также добавить проверку расстояния до 'person' по вертикали (box_center_y) для определения 'назад'

    # Сохраняем последнее изображение и боксы
    last_image = cv_image
    last_boxes = boxes

    return cv_image, direction

# Эндпоинт для загрузки изображения
@app.route('/upload', methods=['POST'])
def upload_image():
    global last_image, last_boxes
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    processed_image, direction = detect_and_draw(image)
    # Возвращаем направление движения в ответе
    response = {
        "status": "Image received and processed",
        "direction": direction
    }
    return jsonify(response)

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
