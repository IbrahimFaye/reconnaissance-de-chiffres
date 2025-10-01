import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Charger le modèle
model = tf.keras.models.load_model('../saved_models/mnist_cnn.h5')

st.title("🎨 Reconnaissance de Chiffres MNIST avec CNN")
st.write("Dessine un chiffre (0-9) dans la zone ci-dessous et le modèle le reconnaîtra.")

# Paramètres du canvas
canvas_size = 280
stroke_width = st.slider("Épaisseur du trait :", 5, 25, 15)

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=stroke_width,
    stroke_color="white",
    background_color="black",
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

# Bouton pour prédire
if st.button("Prédire"):
    if canvas_result.image_data is not None:
        # Récupérer l’image du canvas
        img = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(img).convert("L").resize((28,28))

        # Préparer l'image pour le modèle
        img_array = np.array(img).reshape(1,28,28,1)/255.0

        # Prédiction
        pred = model.predict(img_array)
        predicted_digit = np.argmax(pred)

        st.write(f"### ✅ Le modèle prédit : **{predicted_digit}**")
        st.image(img.resize((140,140)), caption="Image entrée", width=140)
