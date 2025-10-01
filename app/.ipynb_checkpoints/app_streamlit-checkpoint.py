import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

model = tf.keras.models.load_model('../saved_models/mnist_cnn.h5')

st.title("🎨 Reconnaissance de Chiffres MNIST avec CNN")
st.write("Dessine un chiffre (0-9) dans la zone ci-dessous et le modèle le reconnaîtra.")

canvas_size = 280
img = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(img)

# Streamlit canvas
st.write("**Dessinez ici:**")
canvas_data = st.canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=canvas_size,
    width=canvas_size,
    drawing_mode="freedraw",
)

if st.button("Prédire"):
    if canvas_data.image_data is not None:
        img = canvas_data.image_data
        img = Image.fromarray(np.uint8(img[:,:,0])).resize((28,28)).convert("L")
        img_array = np.array(img).reshape(1,28,28,1)/255.0

        pred = model.predict(img_array)
        predicted_digit = np.argmax(pred)

        st.write(f"**Le modèle prédit : {predicted_digit}**")
        st.image(img.resize((140,140)), caption="Image entrée", width=140)
