import torch
import urllib
import streamlit as st
from io import BytesIO
from time import time
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def scale_image(image: Image.Image, target_height: int = 500) -> Image.Image:
    """
    Scale an image to a target height while maintaining the aspect ratio.

    Parameters
    ----------
    image : Image.Image
        The image to scale.
    target_height : int, optional (default=500)
        The target height of the image.

    Returns
    -------
    Image.Image
        The scaled image.
    """
    width, height = image.size
    aspect_ratio = width / height
    target_width = int(aspect_ratio * target_height)
    return image.resize((target_width, target_height))


def upload_image() -> None:
    """
    Upload an image.
    """
    if st.session_state.file_uploader is not None:
        st.session_state.image = Image.open(st.session_state.file_uploader)


def read_image_from_url() -> None:
    """
    Read an image from a URL.
    """
    if st.session_state.image_url is not None:
        with urllib.request.urlopen(st.session_state.image_url) as response:
            st.session_state.image = Image.open(BytesIO(response.read()))


def inference() -> None:
    """
    Perform inference on an image and generate a caption.
    """
    start_time = time()
    outputs = st.session_state.processor(
        images=st.session_state.image,
        return_tensors="pt",
    )
    outputs = {k: v.to(st.session_state.device.lower()) for k, v in outputs.items()}
    st.session_state.model.to(st.session_state.device.lower())
    logits = st.session_state.model.generate(
        **outputs,
        max_length=st.session_state.max_length,
        num_beams=st.session_state.num_beams,
    )
    caption = st.session_state.processor.decode(
        logits[0], skip_special_tokens=True
    )
    end_time = time()

    st.session_state.inference_time = round(end_time - start_time, 2)
    st.session_state.caption = caption

    st.session_state.model.to("cpu")
    torch.cuda.empty_cache()


def main() -> None:
    """
    Main function for the Streamlit app.
    """
    if "model" not in st.session_state:
        st.session_state.model = AutoModelForVision2Seq.from_pretrained(
            "tanthinhdt/blip-base_with-pretrained_flickr30k",
            cache_dir="models/huggingface",
        )
        st.session_state.model.eval()
    if "processor" not in st.session_state:
        st.session_state.processor = AutoProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            cache_dir="models/huggingface",
        )
    if "image" not in st.session_state:
        st.session_state.image = None
    if "caption" not in st.session_state:
        st.session_state.caption = None
    if "inference_time" not in st.session_state:
        st.session_state.inference_time = 0.0

    # Set page configuration
    st.set_page_config(
        page_title="Image Captioning App",
        page_icon="📸",
        initial_sidebar_state="expanded",
    )

    # Set sidebar layout
    st.sidebar.header("Workspace")
    st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        on_change=upload_image,
        key="file_uploader",
        help="Upload an image to generate a caption.",
    )
    st.sidebar.text_input(
        "Image URL",
        on_change=read_image_from_url,
        key="image_url",
        help="Enter the URL of an image to generate a caption.",
    )
    st.sidebar.divider()
    st.sidebar.header("Settings")
    st.sidebar.selectbox(
        label="Device",
        options=["CPU", "CUDA"],
        index=1,
        key="device",
        help="The device to use for inference.",
    )
    st.sidebar.number_input(
        label="Max length",
        min_value=32,
        max_value=128,
        value=64,
        step=1,
        key="max_length",
        help="The maximum length of the generated caption.",
    )
    st.sidebar.number_input(
        label="Number of beams",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        key="num_beams",
        help="The number of beams to use during decoding.",
    )

    # Set main layout
    st.markdown(
        """
        <h1 style='text-align: center;'>
            Image Captioning
        </h1>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    image_container = st.container(height=450)
    st.divider()
    col_1, col_2, col_3 = st.columns([1, 1, 2])
    resolution_display = col_1.empty()
    runtime_display = col_2.empty()
    caption_display = col_3.empty()

    # Display the image and generate a caption
    if st.session_state.image is not None:
        image_container.image(scale_image(st.session_state.image, target_height=400))

        resolution_display.metric(
            label="Image Resolution",
            value=f"{st.session_state.image.width}x{st.session_state.image.height}",
        )

        with st.spinner("Generating caption..."):
            inference()

        caption_display.text_area(
            label="Caption",
            value=st.session_state.caption,
        )
        runtime_display.metric(
            label="Inference Time",
            value=f"{st.session_state.inference_time}s",
        )


if __name__ == "__main__":
    main()
