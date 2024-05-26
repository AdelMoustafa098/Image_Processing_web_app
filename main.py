import streamlit as st
from image_processor import ImageProcessor
import io
import cv2
import numpy as np

st.title("Welcome to Image Processor")

st.header("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the dropdown menu above the images
    st.header("Select Processing Method")
    processing_method = st.selectbox(
        "Choose a processing method",
        (
            "None",
            "Convert to Gray",
            "Add Salt & pepper noise",
            "Add gussian noise",
            "Add uniform noise",
            "Average filter (3x3)",
        ),
    )

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    processor = ImageProcessor(image)

    if processing_method != "None":
        if processing_method == "Convert to Gray":
            processor.convert_to_gry()
        elif processing_method == "Add Salt & pepper noise":
            processor.add_salt_pepper_noise()
        elif processing_method == "Add gussian noise":
            processor.add_gussian_noise()
        elif processing_method == "Add uniform noise":
            processor.add_uniform_noise()
        elif processing_method == "Average filter (3x3)":
            processor.avg_filter()

    else:
        pass

    # Display the images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(
            processor.get_image(),
            caption=f"Processed Image ({processing_method})",
            use_column_width=True,
        )

    # Convert processed image to JPEG format for download
    is_success, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(processor.get_image(), cv2.COLOR_RGB2BGR)
    )
    io_buf = io.BytesIO(buffer)

    # Provide a download button for the processed image
    st.download_button(
        label="Download Processed Image",
        data=io_buf,
        file_name="processed_image.jpg",
        mime="image/jpeg",
    )
