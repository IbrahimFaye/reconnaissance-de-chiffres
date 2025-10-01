import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mnist_cnn.h5")
model = tf.keras.models.load_model(MODEL_PATH)

st.title("🎨 Reconnaissance de Chiffres MNIST avec CNN")
st.write("Dessine un chiffre (0-9) dans la zone ci-dessous et le modèle le reconnaîtra.")

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

if st.button("Prédire"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        img = Image.fromarray(img).convert("L").resize((28,28))

        img_array = np.array(img).reshape(1,28,28,1)/255.0

        pred = model.predict(img_array)
        predicted_digit = np.argmax(pred)
        confidence = np.max(pred)

        threshold = 0.87
        st.write(f"### : **confiance  {confidence:.2%}**")

        if confidence < threshold:
            st.warning("🤔 Je ne suis pas sûr... (confiance  faible) Vous pouvez améliorer  votre écriture svp")
        else:
            st.write(f"### ✅ Le modèle prédit : **{predicted_digit}**")
        st.image(img.resize((140,140)), caption="Image entrée", width=140)
