from skimage.feature import haar_like_feature, hog
from concurrent.futures import ThreadPoolExecutor

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import cv2
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        return np.zeros(8100)
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Улучшает контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)

    features = hog(gray_image,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   orientations=9,
                   block_norm='L2-Hys')
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
        print("Готово - " + str(i))
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
        SVC(kernel='rbf', gamma='scale')
    )

    pipeline.fit(X_train, y_train)

    print(f"Train accuracy: {pipeline.score(X_train, y_train):.2f}")
    print(f"Test accuracy: {pipeline.score(X_test, y_test):.2f}")

    return pipeline



def detect_faces_hog(image, model):
  detections = []
  for (x, y, window) in sliding_window(image):
      features = extract_hog_features(window)
      prediction = model.predict([features])
      if prediction == 1:  # Если предсказано, что это лицо
          detections.append((x, y, x + window.shape[1], y + window.shape[0]))  # Записываем координаты окна

  return detections



def detect_and_draw_hog(path, model, path_to_save):
    # Обнаружение лиц на новом изображении
    test_image = cv2.imread(path)
    detections = detect_faces_hog(test_image, model)
    print(detections)
    # Отрисовка обнаруженных лиц
    output_image = draw_detections(test_image, detections)
    cv2.imwrite(path_to_save, output_image)