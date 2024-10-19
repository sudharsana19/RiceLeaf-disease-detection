import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Load the pre-trained model
model = load_model('cnn_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256,256))  # Resize to the input size expected by your model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict the disease
def predict_disease(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return predictions

# Streamlit app
st.title('Rice Leaf Disease Prediction')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    classes = ['bacterial_leaf_blight', 'brown_spot','leaf_smut']
    
    # Predict
    predictions = predict_disease(image)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 10 # Get confidence percentage


    # Display the prediction
    st.write(f'The predicted Disease is: {predicted_class}')
    st.write(f'Thankyou !...')
    st.write(f"Confidence: {confidence:.2f}%")

    #  outputs probabilities for 3 classes
    classes = ['bacterial_leaf_blight', 'brown_spot','leaf_smut']
    prediction_dict = dict(zip(classes, predictions[0]))


    
    # Display the predictions
    st.write("Prediction probabilities by a bar graph:")
    st.bar_chart(prediction_dict)
    st.write("Probabilities of Each classes:")
    
    # Optionally, display the predictions as text
    st.write(prediction_dict)

    # Additional developer details
st.write('**Built by: Sudharsana Vigneshwaran A**')
st.write('**Linked in** [Sudharsana Vigneshwaran A](https://www.linkedin.com/in/sudharsana-vigneshwaran/)')
st.write('**GitHub:** [Sudharsana Vigneshwaran A](https://github.com/sudharsana19)')