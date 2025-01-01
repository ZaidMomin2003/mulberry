import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Define labels
LABELS = {
    0: "Healthy",
    1: "Leafrust 25-40",
    2: "Leafrust 40-60",
    3: "Leafrust 60-100",
    4: "Leafspot 25-40",
    5: "Leafspot 40-60",
    6: "Leafspot 60-100"
}

# Load the model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Process image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Make prediction
def predict(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    return predicted_label, output_data

# Streamlit App
def main():
    st.title("Mulberry Disease Detection")
    st.write("Upload an image of the Mulberry plant to detect the disease.")

    model = load_model()

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Preprocess and predict
        processed_image = preprocess_image(image, (224, 224))  # Assuming the model expects 224x224 input
        label, probabilities = predict(processed_image, model)

        # Display result
        st.write(f"The detected crop condition is: **{LABELS[label]}**")

if __name__ == "__main__":
    main()
