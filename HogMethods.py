import torch
from skimage.feature import haar_like_feature, hog
from concurrent.futures import ThreadPoolExecutor

from sklearn import svm
from sklearn.model_selection import train_test_split
from torchvision.ops import nms
from sklearn.ensemble import AdaBoostClassifier
import cv2
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from GeneralMethods import load_annotations, sliding_window, draw_detections
import numpy as np
from GeneralMethods import FaceDataset


# Возвращает обученную модель
def get_trained_hog_model(faces_path, non_faces_path):
    # Загрузка изображений и аннотаций
    images, annotations = load_annotations(faces_path, non_faces_path)

    # Подготовка данных
    X, y = prepare_hog_data(images, annotations)

    # Обучение модели AdaBoost
    print("Идет обучение...")
    model = train_svm_based_on_hog(X, y)
    return model


# Функция для извлечения признаков HOG
def extract_hog_features(image):
    if image is None or image.size == 0:
        raise ValueError("Пустое изображение")
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Улучшает контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)

    features = hog(gray_image,
                   pixels_per_cell=(6, 6),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True)
    return features


# Возвращает X: вектор HOG, y: 1, если это лицо, 0 если нет. Генерирует не-лица
def prepare_hog_data(images, annotations):
    X = []
    y = []

    dataset = FaceDataset(images, annotations, extract_hog_features)

    def process_item(idx):
        features, label = dataset[idx]

        return features, label

    i = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        indices = list(range(len(dataset)))
        results = executor.map(process_item, indices)
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def train_svm_based_on_hog(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    pipeline = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=0.01,
            penalty='l2',  # Норма регуляризации
            class_weight='balanced',
            max_iter=10000,
            verbose=1
        )
    )

    pipeline.fit(X_train, y_train)

    print(f"Train accuracy: {pipeline.score(X_train, y_train):.2f}")
    print(f"Test accuracy: {pipeline.score(X_test, y_test):.2f}")

    return pipeline


def detect_faces_hog(image, model, threshold=1.5):
    detections = []
    scores = []

    for (x, y, window) in sliding_window(image):
        features = extract_hog_features(window)
        score = model.decision_function([features])[0]  # Извлекаем коэффициент уверенности
        if score > threshold:
            detections.append([x, y, x + window.shape[1], y + window.shape[0]])

            scores.append(score)

    if len(detections) > 0:
        detections = np.array(detections, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        keep = nms(torch.tensor(detections), torch.tensor(scores), iou_threshold=1)
        print(detections, scores)
        return detections[keep]
    return []


def detect_and_draw_hog(path, model, path_to_save):
    # Обнаружение лиц на новом изображении
    test_image = cv2.imread(path)
    detections = detect_faces_hog(test_image, model)

    # Отрисовка обнаруженных лиц
    output_image = draw_detections(test_image, detections)
    cv2.imwrite(path_to_save, output_image)

def classify_face_hog(path, model):
    test_image = cv2.imread(path)
    features = extract_hog_features(test_image)
    print("Это лицо - " + str(model.decision_function([features])))

def classify_face_hog_by_image(image, model):
    features = extract_hog_features(image)
    print("Это лицо - " + str(model.decision_function([features])))
