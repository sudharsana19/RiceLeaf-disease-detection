import streamlit as st
import numpy as np
import tensorflow
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('cnn_model.h5')

# Class labels
class_labels = ['bacterial_leaf_blight', 'brown_spot','leaf_smut']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize the image to match model input
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array


# Streamlit application
st.title("Rice Leaf Disease Prediction using CNN")
st.write("Upload an image of a rice leaf to predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image.load_img(uploaded_file)
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 10 # Get confidence percentage

    # Display the prediction
    st.write(f'The predicted Disease is: {predicted_class}')
    st.write(f'Thankyou !...')
    st.write(f"Confidence: {confidence:.2f}%")

        # Additional developer details
st.write('**Built by: Sudharsana Vigneshwaran A**')
st.write('**Linked in** [Sudharsana Vigneshwaran A](https://www.linkedin.com/in/sudharsana-vigneshwaran/)')
st.write('**GitHub:** [Sudharsana Vigneshwaran A](https://github.com/sudharsana19)')