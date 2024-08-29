import streamlit as st

st.title("Hellooo")

st.header("Gooddd")

st.subheader("hiii")

st.text("This is a text")

st.markdown("### This is a markdown")

# success
st.success("Success")
 
# success
st.info("Information")
 
# success
st.warning("Warning")
 
# success
st.error("Error")
 
# Exception - This has been added later
exp = ZeroDivisionError("Trying to divide by Zero")
st.exception(exp)

 
# Write text
st.write("Text with write")
 
# Writing python inbuilt function range()
st.write(range(10))

# Display Images
 
# import Image from pillow to open images
from PIL import Image
img = Image.open("im1.png")
 
# display image using streamlit
# width is used to set the width of an image
st.image(img, width=200)