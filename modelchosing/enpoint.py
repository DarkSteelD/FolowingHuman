import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr
import threading

# Инициализируем модель YOLO
model = YOLO("yolov9c.pt")
model.info()

# Глобальные переменные для последнего изображения и боксов
last_image = None
last_boxes = None

# Функция для обработки детекции и центрирования класса 'person'
def detect_and_draw(image):
    global last_image, last_boxes
    cv_image = cv2.resize(image, (1080, 1080))
    
    # Выполняем детекцию объектов
    results = model(cv_image)
    boxes = []
    target_class = 'person'
    target_box = None

    # Проходим по всем обнаруженным объектам
    for box in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Используем box.xyxy для получения координат
        width = x_max - x_min
        height = y_max - y_min
        class_id = int(box.cls[0])  # Получаем индекс класса
        class_name = model.names[class_id]  # Получаем имя класса

        # Если класс - 'person', сохраняем его для центрирования
        if class_name == target_class:
            target_box = (x_min, y_min, x_max, y_max)
            break  # Останавливаемся на первом найденном 'person'
        
        boxes.append({"x": x_min, "y": y_min, "width": width, "height": height})
    
    if target_box:
        x_min, y_min, x_max, y_max = target_box
        box_center_x = (x_min + x_max) // 2
        box_center_y = (y_min + y_max) // 2
        frame_center_x, frame_center_y = cv_image.shape[1] // 2, cv_image.shape[0] // 2

        # Смещение изображения, чтобы 'person' оказался в центре
        shift_x, shift_y = frame_center_x - box_center_x, frame_center_y - box_center_y
        cv_image = cv2.warpAffine(cv_image, np.float32([[1, 0, shift_x], [0, 1, shift_y]]), (1080, 1080))
    
    # Сохраняем последнее изображение и боксы
    last_image = cv_image
    last_boxes = boxes

    # Рисуем боксы на изображении
    for box in boxes:
        x, y, width, height = box['x'], box['y'], box['width'], box['height']
        cv2.rectangle(cv_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return cv_image

# Захват видео с локальной камеры
def capture_from_camera():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обработка кадра для детекции и центрирования
        processed_frame = detect_and_draw(frame)
        
        # Отображаем кадр
        cv2.imshow("YOLO Detection - Centering 'person'", processed_frame)
        
        # Остановка при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Запуск захвата с камеры в отдельном потоке
threading.Thread(target=capture_from_camera).start()
