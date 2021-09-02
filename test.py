# importing necessary modules and packages
import cv2
import tensorflow as tf

# four categories of data 
CATEGORIES = ["adidas", "nike" , "puma" , "reebok"]

# test image directory
test = "C:/Users/NANDITHA/Desktop/B.Tech/S8/AI/Project/data/test_data/adidas.jpg"

# function to get the test image and resize it
def prepare(filepath):
    IMG_SIZE = 100  # resize parameter
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # reading the image as grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # converting to 1D array

# calling the saved model
model = tf.keras.models.load_model("CNN.model")

# prediting using the model
prediction = model.predict([prepare(test)])
print(prediction)  
for i in range(4):
    if (prediction[0][i]==1):
        print(CATEGORIES[i])
