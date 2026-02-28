import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Iris AI System", layout="centered")

# load trained model
model = tf.keras.models.load_model("iris_model.h5")

# -------- STYLE --------
st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#fef9c3,#fde68a);
}

h1,h2,h3,label{
color:#78350f !important;
font-weight:bold;
}

.result{
background:#78350f;
padding:18px;
border-radius:14px;
text-align:center;
font-size:26px;
font-weight:bold;
color:white;
box-shadow:0 4px 20px rgba(0,0,0,0.2);
}

.small{
text-align:center;
font-size:16px;
color:#92400e;
margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# -------- TITLE --------
st.markdown("<h1 style='text-align:center;'>🌼 Iris Flower Prediction</h1>", unsafe_allow_html=True)
st.markdown("<div class='small'>Machine Learning Project</div>", unsafe_allow_html=True)

# -------- INPUTS --------
c1,c2 = st.columns(2)

with c1:
    a = st.number_input("Sepal Length",5.1)
    b = st.number_input("Sepal Width",3.5)

with c2:
    c = st.number_input("Petal Length",1.4)
    d = st.number_input("Petal Width",0.2)

st.write("")

# -------- BUTTON --------
if st.button("Predict Species", use_container_width=True):
    data = np.array([[a,b,c,d]])
    pred = model.predict(data)
    r = np.argmax(pred)

    names=["🌱 Iris Setosa","🌼 Iris Versicolor","🌸 Iris Virginica"]

    st.markdown("### Prediction Result")
    st.markdown(f"<div class='result'>{names[r]}</div>",unsafe_allow_html=True)