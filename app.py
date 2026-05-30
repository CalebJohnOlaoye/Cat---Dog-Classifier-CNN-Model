
import streamlit as st
from keras.models import load_model
import numpy as np 
from keras.preprocessing import image


@st.cache_resource
def load_my_model():
    # looks for the file in the same directory as the app.py file
    return load_model("cat_dog_classifier.keras")

model = load_my_model()

st.title("Cat vs Dog Classifier")

upload = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)
if upload is None:
  st.info("""
    ## Please Upload an IMAGE of your favourite pet
  """)
else:   
    test_image = image.load_img(upload, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    st.write(f"---{result[0][0]}")
    st.subheader('Prediction Result:')
    if result[0][0] >= 0.5:
        prediction = 'DOG'
        st.success(f"The Model predicts this is a **{prediction}** Confidence : {result[0][0]:.2f}!")
    else:
        prediction = 'CAT'
        st.success(f"The Model predicts this is a **{prediction}** Confidence : {result[0][0]:.2f}!")

    
    
   