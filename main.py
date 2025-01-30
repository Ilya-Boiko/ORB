import cv2
import matplotlib.pyplot as plt
import numpy as np

def rotate_image(image, angle):
    # Функция для вращения изображения на заданный угол
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def resize_image(image, scale_percent):
    # Функция для изменения масштаба изображения
    width = int(image.shape[1] * scale_percent  / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def find_object_and_display(object_image_path, scene_image_path):
    # Загружаем изображения в цвете
    object_img = cv2.imread(object_image_path)
    scene_img = cv2.imread(scene_image_path)

    # Вращаем изображение сцены на x градусов
    #scene_img = rotate_image(scene_img, 20)

    # Изменяем масштаб изображения сцены до 70%
    scene_img = resize_image(scene_img, 100)

    # Инициализируем ORB
    orb = cv2.ORB_create()
    #orb = cv2.ORB_create(nfeatures=1500)

    # Находим ключевые точки и дескрипторы
    keypoints_object, descriptors_object = orb.detectAndCompute(object_img, None)
    keypoints_scene, descriptors_scene = orb.detectAndCompute(scene_img, None)

    # Создаем объект для сопоставления дескрипторов
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Сопоставляем дескрипторы
    matches = bf.match(descriptors_object, descriptors_scene)

    # Сортируем совпадения по расстоянию
    matches = sorted(matches, key=lambda x: x.distance)

    # Устанавливаем порог для минимального количества совпадений
    MIN_MATCH_COUNT = 10
    if len(matches) < MIN_MATCH_COUNT:
        print("[w] недостаточно совпадений - %d/%d" % (len(matches), MIN_MATCH_COUNT))
        return

    # Если достаточно совпадений, извлекаем точки
    src_pts = np.float32([keypoints_object[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Находим матрицу преобразования
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Определяем координаты рамки шаблона
    h, w = object_img.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    # Преобразуем координаты рамки
    dst = cv2.perspectiveTransform(pts, M)
    dst = [np.int32(np.abs(dst))]  # обрезаем рамку вылезшую за пределы картинки

    # Рисуем рамку вокруг найденного объекта
    scene_img_with_box = cv2.polylines(scene_img, dst, True, (0, 0, 255), 2, cv2.LINE_AA)

    # Рисуем совпадения
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(255, 0, 0),  # совпадения в красном цвете
                       singlePointColor=None,
                       matchesMask=matchesMask,  # рисуем только инлайеры
                       flags=2)
    matched_img = cv2.drawMatches(object_img, keypoints_object, scene_img_with_box, keypoints_scene, matches, None, **draw_params)

    # Отображаем изображения
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))  # Преобразуем BGR в RGB для корректного отображения
    plt.axis('off')
    plt.show()

# Пример использования
#find_object_and_display('images/cards/sample/GrootGroot.JPG', 'images/cards/main_image/Groot.JPG')
find_object_and_display('images/cards/sample/9.JPG', 'images/cards/main_image/duo.JPG')
find_object_and_display('images/cards/sample/1.JPG', 'images/cards/main_image/cards.JPG')
find_object_and_display('images/cards/sample/2.JPG', 'images/cards/main_image/cards.JPG')
#find_object_and_display('images/cards/sample/6.JPG', 'images/cards/main_image/cats.JPG')
#find_object_and_display('images/cards/sample/person.jpg', 'images/cards/main_image/people3.JPG')
#find_object_and_display('images/cards/sample/bona.png', 'images/cards/main_image/drinks.JPG')
#find_object_and_display('images/cards/sample/car.jpg', 'images/cards/main_image/cars.JPG')