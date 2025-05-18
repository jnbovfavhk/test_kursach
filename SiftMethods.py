import os

import cv2
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from torchvision.ops import nms

from GeneralMethods import load_annotations, sliding_window, draw_detections


class FaceDetector:
    def __init__(self, n_clusters=150, svm_kernel='rbf'):
        """
        Инициализация детектора лиц.

        Параметры:
            n_clusters: количество кластеров для K-means
            svm_kernel: тип ядра для SVM ('linear', 'rbf', etc.)
        """
        self.n_clusters = n_clusters
        self.svm_kernel = svm_kernel
        self.sift = cv2.SIFT.create(
            contrastThreshold=0.02,
        )
        self.kmeans = None
        self.svm = None
        self.scaler = StandardScaler()
        self.pipeline = None

    def _extract_sift_descriptors(self, image):
        """Извлекает SIFT-дескрипторы из изображения"""
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        resized = cv2.resize(gray_image, (300, 300), interpolation=cv2.INTER_AREA)
        kp, desc = self.sift.detectAndCompute(resized, None)
        return desc if desc is not None else np.array([])

    def _create_bovw_vector(self, descriptors):
        """Преобразует дескрипторы в BoVW-вектор"""
        if len(descriptors) == 0:
            return np.zeros(self.n_clusters)

        labels = self.kmeans.predict(descriptors)
        hist = np.bincount(labels, minlength=self.n_clusters)
        return hist / (np.linalg.norm(hist) + 1e-6)  # Нормализация

    #  Собирает дескрипторы для обучения K-means и примеры для классификации
    # Возвращает:
    # all_descriptors: все дескрипторы для обучения K-means
    # samples: список кортежей (descriptors, label) для обучения модели
    def _collect_training_samples(self, image_paths, annotations):

        all_descriptors = []
        samples = []

        for img_path, label in zip(image_paths, annotations):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            desc = self._extract_sift_descriptors(img)
            if len(desc) > 0:
                all_descriptors.extend(desc)
                samples.append((desc, label))

        return np.vstack(all_descriptors), samples

    def train(self, image_paths, annotations):
        """
        Полный цикл обучения:
        1. Извлечение дескрипторов
        2. Обучение K-means
        3. Создание BoVW-векторов
        4. Обучение SVM
        """
        # Сбор всех дескрипторов и примеров
        all_descriptors, samples = self._collect_training_samples(image_paths, annotations)

        # Обучение K-means
        print("Обучение K-means...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=1024, n_init=5, max_iter=300)
        self.kmeans.fit(all_descriptors)

        # Создание BoVW-векторов
        print("Создание BoVW-векторов...")
        X = []
        y = []

        for desc, label in samples:
            bovw = self._create_bovw_vector(desc)
            X.append(bovw)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Обучение SVM
        print("Обучение SVM...")
        self.svm = LinearSVC(
            C=0.01,
            penalty='l2',  # Норма регуляризации
            verbose=1
        )

        # Создание pipeline с нормализацией
        self.pipeline = make_pipeline(
            StandardScaler(),
            self.svm
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.pipeline.fit(X_train, y_train)

        # Оценка
        print(f"Train accuracy: {self.pipeline.score(X_train, y_train):.2f}")
        print(f"Test accuracy: {self.pipeline.score(X_test, y_test):.2f}")

        return self

    # Возвращает true, если лицо есть на изображении и уверенность модели
    def predict_image(self, img, threshold=2):

        if img is None:
            return False, 0.0

        desc = self._extract_sift_descriptors(img)
        if len(desc) == 0:
            return False, 0.0

        bovw = self._create_bovw_vector(desc).reshape(1, -1)
        proba = self.pipeline.decision_function(bovw)[0]

        return proba >= threshold, proba

    def save_model(self, path):
        joblib.dump({
            'kmeans': self.kmeans,
            'pipeline': self.pipeline,
            'n_clusters': self.n_clusters
        }, path)

    @classmethod
    def load_model(cls, path):
        data = joblib.load(path)
        detector = cls(n_clusters=data['n_clusters'])
        detector.kmeans = data['kmeans']
        detector.pipeline = data['pipeline']
        return detector

    # Возвращает cписок прямоугольников лиц в формате (x1, y1, x2, y2)
    def detect_faces(self, image_path):

        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = []
        scores = []

        for (x, y, window) in sliding_window(gray):
            window_is_face, confidence = self.predict_image(window)
            if window_is_face == False:
                continue

            win_h, win_w = window.shape[:2]
            detections.append((x, y, x + win_w, y + win_h))
            scores.append(confidence)
            # rectangles.append((x, y, x + win_w, y + win_h))
            # print((x, y, x + win_w, y + win_h), "confidence: " + str(confidence))
        if len(detections) > 0:
            # Применяем Non-Maximum Suppression
            detections = np.array(detections, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            keep = nms(torch.tensor(detections), torch.tensor(scores), iou_threshold=0.2)
            print(detections, scores)
            return detections[keep]
        return[]

    def detect_and_draw(self, img_path, path_to_save):

        detections = self.detect_faces(img_path)

        img = cv2.imread(img_path)
        result_img = draw_detections(img, detections)

        # Создаем директорию, если ее нет
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)

        # Сохраняем результат
        if not cv2.imwrite(path_to_save, result_img):
            print(f"Ошибка: не удалось сохранить изображение {path_to_save}")
            return False

        print(f"Обнаружено лиц: {len(detections)}. Результат сохранен в {path_to_save}")
        return True


def get_trained_sift_model(images_path, labels_path):
    img_paths, annotations = load_annotations(images_path, labels_path)
    model = FaceDetector()
    model.train(img_paths, annotations)
    return model
