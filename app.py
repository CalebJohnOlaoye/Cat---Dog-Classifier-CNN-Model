
import streamlit as st
from keras.models import load_model
import numpy as np 
from keras.preprocessing import image

model = load_model("cat_dog_classifier.keras")

st.title("Cat vs Dog Classifier")

upload = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)
if upload == None:
  st.write("""
    ## Please Upload an IMAGE of your favourite pet
  """)
else:   
    test_image = image.load_img(upload, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] >= 0.5:
        prediction = 'DOG'
    else:
        prediction = 'CAT'

    st.write(f"---{result[0][0]}")
    st.subheader('Prediction Result:')
    st.success(f"The Model predicts this is a **{prediction}**!")