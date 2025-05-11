import os
import random
import shutil
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2


def reduct_images_dataset(faces_path, images_path, new_size):
    # Получаем список всех имен файлов в указанной папке
    files_faces_path = [f for f in os.listdir(faces_path)]
    files_images_path = [f for f in os.listdir(images_path)]

    # Проверяем, достаточно ли файлов в папке
    if len(files_faces_path) < new_size or len(files_images_path) < new_size:
        raise ValueError("Недостаточно файлов в папке для выбора.")

    # Случайно выбираем указанное количество файлов из каждой папки
    selected_files1 = random.sample(files_faces_path, new_size)
    selected_files2 = random.sample(files_images_path, new_size)

    reducted_faces_folder = 'reducted/faces'
    reducted_images_folder = 'reducted/images'

    os.makedirs(reducted_faces_folder, exist_ok=True)
    os.makedirs(reducted_images_folder, exist_ok=True)

    # Удаляем всё, что есть в папках
    for item in os.listdir(reducted_faces_folder):
        item_path = os.path.join(reducted_faces_folder, item)
        os.remove(item_path)

    for item in os.listdir(reducted_images_folder):
        item_path = os.path.join(reducted_images_folder, item)
        os.remove(item_path)

    # Копируем выбранные файлы в папку назначения
    for file_name in selected_files1:
        shutil.copy(os.path.join(faces_path, file_name), os.path.join(reducted_faces_folder, file_name))

    for file_name in selected_files2:
        shutil.copy(os.path.join(images_path, file_name), os.path.join(reducted_images_folder, file_name))
    print("Сокарщенный датасет создан")
    print("Количество тренировочных единиц в данных - " + str(new_size))
    return reducted_faces_folder, reducted_images_folder


# Возвращает список путей к изображениям и список меток к ним
def load_annotations(faces_dir, non_faces_dir):
    image_paths = []
    labels = []
    for f in os.listdir(faces_dir):
        image_paths.append(os.path.join(faces_dir, f))
        labels.append(1)

    for f in os.listdir(non_faces_dir):
        image_paths.append(os.path.join(non_faces_dir, f))
        labels.append(0)

    # Перемешиваем списки, но так, чтобы не потерять верные метки
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    return image_paths, labels


class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, extract_features_func):
        self.data = []
        self.extract_features_func = extract_features_func
        for i in range(len(image_paths)):
            self.data.append((image_paths[i], labels[i]))

        print("FaceDataset инициализирован. длина массива данных: " + str(len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = np.array(Image.open(img_path))
        if img.size == 0:
            raise ValueError(f"Пустое изображение: {img_path}")
        if img is None:
            raise RuntimeError(f"Не удалось загрузить изображение {img_path}")

        print("getitem сработал. Индекс: " + str(idx))
        features = self.extract_features_func(img)

        return features, label

    def get_batch(self, indices):
        batch_images = []
        batch_labels = []
        for idx in indices:
            img_path, label = self.data[idx]
            img = np.array(Image.open(img_path))
            batch_images.append(img)
            batch_labels.append(label)

        # Векторизованная обработка
        features_batch = self.extract_features_func(batch_images)
        print(f"Для {len(indices)} индексов извлеклись признаки")
        return features_batch, batch_labels


def sliding_window(image, min_window_size=None, max_window_size=None, aspect_ratio=(1, 2)):
    if max_window_size is None:
        max_window_size = (
            image.shape[1], image.shape[0])  # Установить максимальный размер окна равным размеру изображения

    if min_window_size is None:
        min_window_size = (image.shape[1] // 10, image.shape[0] // 10)

    # if step_size is None:
    #     step_size = min_window_size[0]

    for window_height in range(min_window_size[1], max_window_size[1] + 1,
                               max_window_size[1] // 20):  # Увеличиваем высоту окна
        for window_width in range(min_window_size[0], max_window_size[0] + 1,
                                  max_window_size[0] // 20):  # Увеличиваем ширину окна

            # Находим соотношение сторон
            width_to_height_ratio = window_width / window_height
            height_to_width_ratio = window_height / window_width

            step_size = window_width
            # Проверяем, что хотя бы одно из соотношений в заданном интервале(где может находится лицо)
            if (aspect_ratio[0] <= width_to_height_ratio <= aspect_ratio[1]) or \
                    (aspect_ratio[0] <= height_to_width_ratio <= aspect_ratio[1]):
                for y in range(0, image.shape[0] - window_height + 1, step_size):
                    for x in range(0, image.shape[1] - window_width + 1, step_size):
                        print(f'x = {x}, y = {y}, width = {window_width}, height = {window_height}')
                        yield (x, y, image[y:y + window_height, x:x + window_width])


def draw_detections(image, detections):
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Рисуем прямоугольник
    return image
