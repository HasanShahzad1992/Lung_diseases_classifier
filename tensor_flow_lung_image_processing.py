import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models,layers,datasets
import joblib

opening_main=os.listdir("Lungs_dataset")
all_images=[]
all_labels=[]
for i in opening_main:
    sub_folder_path=os.path.join("Lungs_dataset",i)
    sub_folder_open=os.listdir(sub_folder_path)
    for j in sub_folder_open:
        image_path=os.path.join("Lungs_dataset",i,j)
        image_open=image.load_img(image_path,target_size=(150,150))
        image_array=image.img_to_array(image_open)
        normalized_array=image_array/255
        all_images.append(normalized_array)
        all_labels.append(i)
numpy_images=np.array(all_images)
numpy_labels=np.array(all_labels)
label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(numpy_labels)
joblib.dump(label_encoder,"lung_disease_label_encoder.joblib")
number_of_columns=len(label_encoder.classes_)
one_hot_encoded_table=to_categorical(encoded_labels,number_of_columns)
X=numpy_images
Y=one_hot_encoded_table
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),layers.MaxPool2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Conv2D(128,(3,3),activation="relu"),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dropout(0.2),layers.Dense(number_of_columns,activation="softmax")])
model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=10)
model.save("lungs_disease_classifier.keras")
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)
print(test_loss,test_accuracy)

loaded_model=models.load_model("lungs_disease_classifier.keras")
image_variable=image.load_img("lung_disease.jpg",target_size=(150,150))
image_array=image.img_to_array(image_variable)
expand_dimensions=np.expand_dims(image_array,axis=0)
prediction=loaded_model.predict(expand_dimensions)
predict_digit=np.argmax(prediction)
label_encoder=joblib.load("lung_disease_label_encoder.joblib")
predicted_label=label_encoder.classes_[predict_digit]
print(predicted_label)

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers,models,datasets
import joblib

all_images=[]
all_labels=[]
opening_main=os.listdir("train_dataset")
for i in opening_main:
    sub_folder_path=os.path.join("train_dataset",i)
    opening_sub_folders=os.listdir(sub_folder_path)
    for j in opening_sub_folders:
        image_path=os.path.join("train_dataset",i,j)
        image_open=image.load_img(image_path,target_size=(150,150))
        image_array=image.img_to_array(image_open)
        normalized_array=image_array/255
        all_images.append(normalized_array)
        all_labels.append(i)
numpy_image=np.array(all_images)
numpy_labels=np.array(all_labels)
label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(all_labels)
joblib.dump(label_encoder,"lung_disease_label_encoder.joblib")
number_of_columns=len(label_encoder.classes_)
table_one_hot_encoded=to_categorical(encoded_labels,number_of_columns)
X=numpy_image
Y=table_one_hot_encoded
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)),layers.MaxPool2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Conv2D(128,(3,3),activation="relu"),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dropout(0.2),layers.Dense(number_of_columns,activation="softmax")])
model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=10,batch_size=16)
model.save("lungs_disease_classifier.keras")
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)
print(test_loss,test_accuracy)
loaded_model=models.load_model("lungs_disease_classifier.keras")
image_variable=image.load_img("lung_disease.jpg",target_size=(150,150))
image_array=image.img_to_array(image_variable)
expand_dimensions=np.expand_dims(image_array,axis=0)
prediction=loaded_model.predict(expand_dimensions)
predict_digit=np.argmax(prediction)
label_encoder=joblib.load("lung_disease_label_encoder.joblib")
predict_digit=label_encoder.classes_[predict_digit]
print(predict_digit)
