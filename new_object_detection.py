"""
Created by Anurag at 19-12-2021
"""
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pickle
import cv2
import os

new_csv = pd.read_csv('annotations_train.csv', header=None)
# valid_csv = pd.read_csv('annots_valid.csv', header=None)

data = []
bboxes = []
labels = []
imagePaths = []

for i in range(new_csv.shape[0]):
    image_path = new_csv.iloc[i, 0]
    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    # initializing starting point
    startX = float(new_csv.iloc[i, 1]) / w
    startY = float(new_csv.iloc[i, 2]) / h
    # Also initialize ending point
    endX = float(new_csv.iloc[i, 3]) / w
    endY = float(new_csv.iloc[i, 4]) / h
    # load image and give them default size
    image = load_img(image_path, target_size=(224, 224, 3))
    # see here if we cant take it into float then we face trouble
    image = img_to_array(image)
    # Lets append into data , targets ,filenames
    bboxes.append((startX, startY, endX, endY))
    imagePaths.append(image_path)
    labels.append(new_csv.iloc[i, 5])
    data.append(image)

# for i in range(valid_csv.shape[0]):
#     image_paths_valid = valid_csv.iloc[i, 0]
#     img = cv2.imread(image_paths_valid)
#     (h, w) = img.shape[:2]
#     # initializing starting point
#     # Why we take in float because when we convert into array so then will trouble happen
#     startX = float(valid_csv.iloc[i, 1]) / w
#     startY = float(valid_csv.iloc[i, 2]) / h
#     # Also initialize ending point
#     endX = float(valid_csv.iloc[i, 3]) / w
#     endY = float(valid_csv.iloc[i, 4]) / h
#     # load image and give them default size
#     image = load_img(image_paths_valid, target_size=(224, 224))
#     # see here if we cant take it into float then we face trouble
#     image = img_to_array(image)
#
#     # Lets append into data , targets ,filenames
#     valid_targets.append((startX, startY, endX, endY))
#     valid_filenames.append(image_paths_valid)
#     valid_label.append(valid_csv.iloc[i, 5])
#     valid_data.append(image)

# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range
# [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)
# train_data = np.array(train_data, dtype='float32') / 255.0
# train_targets = np.array(train_targets, dtype='float32')
# train_label = np.array(train_label, dtype='float32')
# valid_label = np.array(valid_label, dtype='float32')
# valid_data = np.array(valid_data, dtype='float32') / 255.0
# valid_targets = np.array(valid_targets, dtype='float32')

# perform one-hot encoding on the labels

# defining binary labeler
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

split = train_test_split(data, labels, bboxes, imagePaths,
                         test_size=0.20, random_state=42)
# unpack the data split
(train_images, test_images) = split[:2]
(train_labels, test_labels) = split[2:4]
(train_bboxes, test__bboxes) = split[4:6]

################################################################################
# Model making and training
################################################################################
vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
print("VGG_ summary : ", vgg.summary())
vgg.trainable = False

flatten = vgg.output

flatten = Flatten()(flatten)

# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid",
                 name="bounding_box")(bboxHead)

# construct a second fully-connected layer head, this one to predict the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(37, activation="softmax",
                    name="class_label")(softmaxHead)

# putting together both model

model = Model(inputs=vgg.input, outputs=(bboxHead, softmaxHead))

print("model summary : ", model.summary())

# define a dictionary that specifies loss for both
losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

# define a dictionary that specifies both loss have equal weight
lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

# initialize the optimizer
opt = Adam(lr=1e-4)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

# construct a dictionary for our target training outputs
trainTargets = {
    "class_label": train_labels,
    "bounding_box": train_bboxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
    "class_label": test_labels,
    "bounding_box": test__bboxes
}
# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
history = model.fit(
    train_images, trainTargets,
    validation_data=(test_images, testTargets),
    batch_size=32,
    epochs=50,
    verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save('best_model' + '.h5')

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# save binarizer for test
print("[INFO] saving label binarizer...")
f = open('label_info', "wb")
f.write(pickle.dumps(lb))
f.close()
