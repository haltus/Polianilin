import cv2
import numpy as np
import matplotlib.pyplot as plt


def countPoints(path):
    # Загружаем изображение
    image_path = path
    image = cv2.imread(image_path)
    const = 7.8125**2
    # Проверяем, загрузилось ли изображение
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Увеличиваем контраст с помощью equalizeHist
    equalized = cv2.equalizeHist(gray)

    # Применяем адаптивную бинаризацию
    adaptive_thresh = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Морфологическая операция для удаления мелкого шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Поиск контуров
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по размеру (пороговая фильтрация)
    min_area = 10  # Минимальная площадь для точек
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    # Вычисляем площади всех контуров
    areas = np.array([cv2.contourArea(cnt) for cnt in filtered_contours])

    # 6. Автоматический расчет порогов для классификации
    small_threshold = np.percentile(areas, 33) + 5  # Порог для маленьких точек
    medium_threshold = np.percentile(areas, 66) + 5  # Порог для средних точек

    # Категоризация точек по размеру
    small_points = [
        cnt for cnt in filtered_contours if cv2.contourArea(cnt) < small_threshold
    ]
    medium_points = [
        cnt
        for cnt in filtered_contours
        if small_threshold <= cv2.contourArea(cnt) < medium_threshold
    ]
    large_points = [
        cnt for cnt in filtered_contours if cv2.contourArea(cnt) >= medium_threshold
    ]

    # Подсчет средней площади для каждой категории
    avg_area_small = (
        np.mean([cv2.contourArea(cnt) for cnt in small_points]) if small_points else 0
    )
    avg_area_medium = (
        np.mean([cv2.contourArea(cnt) for cnt in medium_points]) if medium_points else 0
    )
    avg_area_large = (
        np.mean([cv2.contourArea(cnt) for cnt in large_points]) if large_points else 0
    )

    # Создаем копию изображения для визуализации
    annotated_image = image.copy()
    cv2.drawContours(annotated_image, filtered_contours, -1, (0, 255, 0), 1)
    print(
        f"Маленькие точки: {len(small_points)} со средней площадью {avg_area_small*const} * 10^-8 мм²"
    )
    print(
        f"Средние точки: {len(medium_points)} со средней площадью {avg_area_medium*const} * 10^-8 мм²"
    )
    print(
        f"Большие точки: {len(large_points)} со средней площадью {avg_area_large*const} * 10^-8 мм²"
    )

    return len(small_points), len(medium_points), len(large_points)


small_points = 0
medium_points = 0
large_points = 0
for i in range(29, 30):
    small, medium, large = countPoints(f"EditImage/{i}.jpg") #Счет каждой фотографии
    small_points += small
    medium_points += medium
    large_points += large
# # Вывод результатов
print()
print()
print()
print()
print()
print(f"Маленькие точки: {small_points}")
print(f"Средние точки: {medium_points}")
print(f"Большие точки: {large_points} ")

# График
s = [small_points, medium_points, large_points]
x = range(len(s))
ax = plt.gca()
ax.bar(x, s, align="edge")  # align='edge' - выравнивание по границе, а не по центру
ax.set_xticks(x)
ax.set_xticklabels(("small", "middle", "big"))
plt.show()

# # Вывод порогов
# print(f"Порог для маленьких точек: {small_threshold:.2f} пикселей²")
# print(f"Порог для средних точек: {medium_threshold:.2f} пикселей²")