import shutil
import os
import random
import joblib
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from skimage.feature import haar_like_feature
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import cv2

import SiftMethods
from GeneralMethods import reduct_images_dataset
from HaarMethods import get_trained_haar_model, detect_and_draw_haar, classify_face_haar
from HogMethods import get_trained_hog_model, detect_and_draw_hog
from SiftMethods import get_trained_sift_model


def demonstrate_haar():
    images_path, labels_path = reduct_images_dataset("celebA_and_dtd(textures)/faces",
                                                     "celebA_and_dtd(textures)/images", 1000)

    if not os.path.isfile('models/haar_trained.pkl'):
        model = get_trained_haar_model(images_path, labels_path)
        joblib.dump(model, 'models/haar_trained.pkl')
        print("Модель, основанная на каскадах Хаара, сериализована")
    else:
        model = joblib.load('models/haar_trained.pkl')
        print("Модель десериализована")

    detect_and_draw_haar("testImages/tests/test2_1.jpg",
                         model, "testImages/haarTests/test2_1Haar2000.jpg")
    detect_and_draw_haar("testImages/tests/test2_2.jpg",
                         model, "testImages/haarTests/test2_2Haar2000.jpg")
    detect_and_draw_haar("testImages/tests/test2_3.jpg",
                         model, "testImages/haarTests/test2_3Haar2000.jpg")

    print("Обнаружены лица на тестовом изображении")


def demonstrate_hog():
    images_path, labels_path = reduct_images_dataset("celebA_and_dtd(textures)/faces",
                                                     "celebA_and_dtd(textures)/images", 5000)

    if not os.path.isfile('models/hog_trained.pkl'):
        model = get_trained_hog_model(images_path, labels_path)
        joblib.dump(model, 'models/hog_trained.pkl')
        print("Модель, основанная на HOG, сериализована")
    else:
        model = joblib.load('models/hog_trained.pkl')
        print("Модель десериализована")



    print("Обнаружены лица на тестовом изображении")


def demonstrate_sift():
    images_path, labels_path = reduct_images_dataset("celebA_and_dtd(textures)/faces",
                                                     "celebA_and_dtd(textures)/images", 5000)
    if not os.path.isfile('models/sift_trained.pkl'):
        model = get_trained_sift_model(images_path, labels_path)
        model.save_model('models/sift_trained.pkl')
        print("Модель, основанная на SIFT, сериализована")
    else:
        model = SiftMethods.FaceDetector.load_model('models/sift_trained.pkl')
        print("Модель десериализована")




if __name__ == "__main__":
    demonstrate_haar()
