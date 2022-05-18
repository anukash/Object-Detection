"""
Created by Anurag at 20-12-2021
"""

# import pickle
# lb = pickle.loads(open('label_info', "rb").read())
# print(lb.classes_)
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2

model = load_model('best_model.h5')
lb = pickle.loads(open('label_info', "rb").read())

# img = cv2.imread('chihuahua_158.jpg')
image_path = 'basset_hound_82.jpg'
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image, axis=0)

# model predict
(box_preds, label_preds) = model.predict(image)
(start_x, start_y, endX, endY) = box_preds[0]

# calculating the class label with the largest predicted
i = np.argmax(label_preds, axis=1)
label = lb.classes_[i][0]
print("label : ", label)


image = cv2.imread(image_path)
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

# scale the predicted bounding box
start_x = int(start_x * w)
start_y = int(start_y * h)
endX = int(endX * w)
endY = int(endY * h)

# draw the predicted bounding box and class label on the image
y = start_y - 10 if start_y - 10 > 10 else start_y + 10

cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.rectangle(image, (start_x, start_y), (endX, endY), (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
