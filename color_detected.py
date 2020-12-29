import cv2
import numpy as np


# Диапазон значений rgb для зеленого цвета
MIN_RGB = np.array((0, 112, 0), np.uint8)
MAX_RGB = np.array((115, 255, 115), np.uint8)

IMAGE_PATH = 'bird.jpg'

# Текст для каждого прямоугольника
RECTANGLE_TEXT = 'Green'


def detect_objects_by_color(min_rgb, max_rgb, rectangle_text):
    # while чтобы картинка не закрывалась
    while True:
        # Получаем картинку
        image = cv2.imread(IMAGE_PATH)

        # Меняем цветовое пространство картинки из BGR в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Накладываем на кадр цветовой фильтр
        # для выделения цвета в заданном диапазоне.
        #
        # Зеленые элементы фото становятся белыми,
        # остальные - черными.
        filtered_image = cv2.inRange(rgb_image, min_rgb, max_rgb)

        # Массив из единиц для расширения границ отфильтрованных объектов
        # для большей точности при выделении зеленых объектов
        dilate_array = np.ones((5, 5), 'uint8')

        # Увеличиваем размеры границ отфильтрованных объектов
        # для большей точности
        filtered_image = cv2.dilate(filtered_image, dilate_array)

        # Находим все контуры в кадре
        # Кадр | режим группировки | метод упаковки
        contours, hierarchy = cv2.findContours(
            filtered_image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Отрисовываем прямоугольники для контуров
        draw_rectangles(contours, image, rectangle_text)

        cv2.imshow("Detected image", image)

        # Завершение процесса по нажатию Enter
        # 13 - код клавиши Enter
        if cv2.waitKey(1) == 13:
            break


# Рисование прямоугольников по контуру
def draw_rectangles(contours, image, rectangle_text):
    for contour in contours:
        # Площадь каждого контура
        area = cv2.contourArea(contour)

        # Большие площади окружаем прямоугольником
        if(area > 300):
            # Получаем размеры контура
            # x, y - Левая верхняя точка прямоугольника
            x, y, weight, height = cv2.boundingRect(contour)

            # Создаем прямоугольник по контуру
            # Картинка | вершина | противоположная вершина | цвет | толщина
            rectangle = cv2.rectangle(
                image, (x, y),
                (x + weight, y + height), (0, 0, 0), 2
            )

            # Подпись для прямоугольника
            # Фрагмент | текст | координаты | шрифт | размер | цвет текста
            cv2.putText(
                rectangle, rectangle_text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0)
            )


def main():
    detect_objects_by_color(MIN_RGB, MAX_RGB, RECTANGLE_TEXT)


if __name__ == '__main__':
    main()
