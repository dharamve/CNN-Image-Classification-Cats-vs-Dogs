import tensorflow as tf
import numpy as np

MODEL_PATH = "models/cats_dogs_cnn.keras"

IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

image_path = input("Enter image path: ")

image = tf.keras.utils.load_img(
    image_path,
    target_size=(IMG_SIZE, IMG_SIZE)
)

image_array = tf.keras.utils.img_to_array(image)

image_array = image_array / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)

prediction = model.predict(image_array)[0][0]

if prediction >= 0.5:
    print("Prediction: DOG")
    print("Confidence:", round(prediction * 100, 2), "%")
else:
    print("Prediction: CAT")
    print(
        "Confidence:",
        round((1 - prediction) * 100, 2),
        "%"
    )
