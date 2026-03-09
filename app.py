import streamlit as st
import numpy as np
from PIL import Image
from Setup import find_comparison
import cv2


st.set_page_config(page_title = "Foundation Matcher AI", layout="wide")
if 'my_results' not in st.session_state:
    st.session_state.my_results = ""

def reset_results():
    st.session_state.my_results="waiting"


with st.sidebar:
    st.title("Image Center")
    uploaded_file = st.file_uploader("Upload a selfie...",
                                     type=["jpg", "jpeg", "png"],
                                    on_change=reset_results)

    if uploaded_file is not None:
        file_byte = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_byte, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Source Photo', use_container_width=True)


st.title("Foundation Matcher - Find Your Color!")

if uploaded_file is not None:

    col1, col2 = st.columns([1,2], gap = "large")

    with col1:
        if st.button("Find Match"):
            with st.spinner('Analyzing skin tone...'):
                st.session_state.my_results = find_comparison(image_rgb)

    with col2:
        st.subheader("Match Results:")

        if st.session_state.my_results is None:
            st.error("Face not detected. Please try a clearer photo.")

        elif isinstance(st.session_state.my_results, str):
            st.info("Upload a photo and click 'Find Match' to see results here.")
        
        else:
            st.success("We Found Your Shades!")

            for index, row in st.session_state.my_results.iterrows():
                res_col1, res_col2 = st.columns([1,4])

                with res_col1:
                    hex_color = row['hex']
                    st.markdown(
                        f"""
                        <div style="
                            background-color: {hex_color};
                            height:50px;
                            width:50px;
                            border-radius:10px;
                            border: 2px solid #f0f2f6;
                            box-shadow: 2px 2px 2px rgba(0,0,0,0.1);
                        "></div>
                        """,
                        unsafe_allow_html=True
                    )

                with res_col2:
                    st.write(f"**{row['brand']}**")
                    st.write(f"{row['product']} - *{row['name']}*")

                st.divider()

