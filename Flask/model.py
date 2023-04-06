from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


model = load_model(r"Flask\shapes.h5") # loading our model

def predict(InputImg):
    
    img = image.load_img(InputImg, color_mode='grayscale',target_size = (64,64))
    x = image.img_to_array(img) #converting image to array
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # Add an extra dimension to the input data
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction)
    index= {0: 'circle',
        1: 'square',
        2: 'triangle'}
    result=str(index[predicted_class])
    print(f"Key: {predicted_class}")
    print(f"Result: {result}")

    return result

if __name__ == "__main__":
    result = predict(r"C:\Users\gohee\Desktop\PentaAI\Dataset\dataset\dataset\test\square\square-2497.jpg")
    print(result)