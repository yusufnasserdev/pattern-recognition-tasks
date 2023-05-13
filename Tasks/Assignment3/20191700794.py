import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from joblib import Parallel, delayed
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = 'dataset\\train'
TEST_DIR = 'dataset\\test'

IMG_SIZE_W = 28
IMG_SIZE_H = 28

from PIL import Image
import os

def get_data(DIR):
    images = []
    labels = []

    for img_dir in os.listdir(DIR):
        img_label = img_dir
        img_dir = os.path.join(DIR, img_dir)
        
        for img in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img)
            
            images.append(img_path)
            labels.append(img_label)

    return images, labels

def extract_image_features(image_path):
    image = cv2.imread(image_path)

    # Resize the image to the target
    resized_image = cv2.resize(image, (IMG_SIZE_W, IMG_SIZE_H))

    # get hog representation of image
    observation = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, channel_axis=2)

    return observation

def extract_dataset_from_images(images):
# Since we do the same process on all the images we can parellize to save time
    dataset = Parallel(n_jobs=8)(delayed(extract_image_features)(image_path) for image_path in tqdm(images))
    return dataset

def image_augmentation(folder_path):
    # Create a data generator
    datagen = ImageDataGenerator(
        rotation_range=20,  # rotate the image by up to 10 degrees
        width_shift_range=0.1,  # shift the image horizontally by up to 10% of the width
        height_shift_range=0.1,  # shift the image vertically by up to 10% of the height
        shear_range=0.1,  # apply shear transformation to the image
        zoom_range=0.1,  # zoom in or out of the image by up to 10%
        horizontal_flip=True,  # flip the image horizontally
        vertical_flip=True,  # flip the image vertically
        fill_mode='nearest'  # fill in any empty pixels with the nearest value
    )

    for img_dir in os.listdir(folder_path):
        img_dir = os.path.join(folder_path, img_dir)
        
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(img_dir, filename)
                img = cv2.imread(img_path)

                # Reshape the image
                img = img.reshape((1, ) + img.shape)

                # Generate 5 augmented images
                i = 0
                for batch in datagen.flow(img, batch_size=1, save_to_dir=img_dir, save_prefix='aug', save_format='jpg'):
                    i += 1
                    if i > 50:
                        break

def delete_augmented_images(folder_path):
    for img_dir in os.listdir(folder_path):
        img_dir = os.path.join(folder_path, img_dir)
        
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") and filename.startswith("aug"):
                os.remove(os.path.join(img_dir, filename))

# Delete the augmented images remaining from the previous run
delete_augmented_images(TRAIN_DIR)

# Augment the training data
image_augmentation(TRAIN_DIR)

images, labels = get_data(TRAIN_DIR)

X = np.array(extract_dataset_from_images(images))
y = np.array(labels)

# Delete the augmented images after extracting the features
delete_augmented_images(TRAIN_DIR)

print(X.shape)

# Create the model
model = svm.SVC(kernel='rbf', C=2, gamma='scale')

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

scaler = preprocessing.StandardScaler()

# Split the data into k folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop over the folds
for train_index, test_index in skf.split(X, y):
    # Get the current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Scale the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train and evaluate the model on the current fold
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Fold score: {score}")
    print(confusion_matrix(y_test, model.predict(X_test)))
    print('--------------------------------------')

test_imgs, test_labels = get_data(TEST_DIR)

# Extract the features from the test data
test_X = np.array(extract_dataset_from_images(test_imgs))
test_X = scaler.transform(test_X)

# Encode the labels
test_y = np.array(test_labels)
test_y = le.transform(test_y)

# Predict the test data
y_pred = model.predict(test_X)

# Print the accuracy
print(accuracy_score(test_y, y_pred))

# Print the confusion matrix
print(confusion_matrix(test_y, y_pred))

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(test_y, y_pred)

# Define the class labels
class_names = ['accordian', 'ball', 'dollar', 'bike']

# Create the plot
fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(8,8), class_names=class_names, show_normed=True)
plt.title('Confusion Matrix')
plt.show()
