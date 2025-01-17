import keras
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.models import Sequential
from keras.applications import MobileNetV2, ResNet152, VGG16, EfficientNetB0, InceptionV3
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
#from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

#def focal_loss(alpha=0.25, gamma=2.0):
#    def focal_loss_fixed(y_true, y_pred):
#        y_true = tf.cast(y_true, tf.float32)
#        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Avoid numerical instability
#        cross_entropy = -y_true * tf.math.log(y_pred)
#        weight = alpha * tf.math.pow(1 - y_pred, gamma)
#        return tf.reduce_sum(weight * cross_entropy, axis=-1)
#    return focal_loss_fixed

import tensorflow as tf
from tensorflow.keras import backend as K
# from your_module import focal_loss

# def focal_loss(alpha=0.25, gamma=2.0):
#     def loss(y_true, y_pred):
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         cross_entropy = -y_true * tf.math.log(y_pred)
#         weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
#         return K.sum(weight * cross_entropy, axis=-1)
#     return loss

# model.compile(optimizer='adam', loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])








def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(236, 236))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features

TRAIN_DIR = "/kaggle/input/induct/Data/Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

train_features = extract_features(train['image'])

x_train = train_features / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)

model = Sequential()
# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))




model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(2, activation='softmax'))
# model.compile(optimizer='adam', loss=focal_loss(alpha=0.25, gamma=2.0), metrics=['accuracy'])
model.compile(optimizer=RMSprop(learning_rate=0.0032), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(x=x_train, y=y_train, batch_size=15, epochs=25,callbacks=[early_stopping])#new addition teasting to prevent overfitting apperently




import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img

# Example directory for test images
TEST_DIR = "/kaggle/input/abcdef/Test_Images"
test_image_paths = [os.path.join(TEST_DIR, img) for img in os.listdir(TEST_DIR)]

# Process test images
def preprocess_test_images(image_paths):
    features = []
    ids = []
    for image_path in image_paths:
        try: 
            img = load_img(image_path, target_size=(236, 236))
            img_array = np.array(img) / 255.0  # Normalize
            features.append(img_array)
            ids.append(os.path.basename(image_path))  # Save image IDs
        except Exception as e:
            print(f"Skipping file {image_path}: {e}")
    return np.array(features), ids

test_features, test_ids = preprocess_test_images(test_image_paths)
predictions = model.predict(test_features)
predicted_labels = np.argmax(predictions, axis=1)

# Map numeric predictions back to class names
class_names = le.classes_  # Assuming the LabelEncoder used in training is available
mapped_labels = [class_names[label] for label in predicted_labels]

# Create a DataFrame for submission
submission = pd.DataFrame({
    "Id": [img.split('.')[0] for img in test_ids],  # Remove file extensions for IDs
    "Label": mapped_labels
})

# Save to CSV
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")