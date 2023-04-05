from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image


model = load_model(r"Flask\shapes.h5") # loading our model

def predict(InputImg):
    
    img = image.load_img(InputImg, color_mode='grayscale',target_size = (64,64))
    x = image.img_to_array(img) #converting image to array
    x = np.expand_dims(x, axis=0)  # Add an extra dimension to the input data
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction)
    index= {0: 'circle',
        1: 'rhombus',
        2: 'square',
        3: 'trapezoid',
        4: 'triangle'}
    print(f"Key: {predicted_class}")
    result=str(index[predicted_class])
    return result

if __name__ == "__main__":
    result = predict(r"Dataset\dataset\\test\\trapezoid\\trapezoid-2000.jpg")
    print(result)