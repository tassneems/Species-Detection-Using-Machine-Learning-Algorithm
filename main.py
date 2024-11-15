import streamlit as st
import tensorflow as tf
import numpy as np
import time
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

st.set_page_config(
    page_title="Bird Species Detection",  
    page_icon="🦜",  
    layout="centered",
)

def load_model(model_dir):
    """Load saved model and build the detection function."""
    print('Loading model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(model_dir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

def load_label_map(path_to_labels):
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    return category_index

def load_image_into_numpy_array(image):
    return np.array(image)

def run_inference(detect_fn, image_np, category_index, thresh=0.75):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(int)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=thresh,
        agnostic_mode=False)

    return image_np_with_detections, detections

def display_species_info(species_id):
    """Display detailed information for each detected species."""
    species_info = {
        1: {
            "name": "Erithacus rubecula (European Robin)",
            "description": "The European Robin is a small insectivorous songbird commonly found in Europe. It's recognized by its red breast and is a popular bird in gardens.",
            "habitat": "Woodland, gardens, and parks throughout Europe.",
            "conservation_status": "Least Concern",
            "fun_fact": "European robins are known to be very territorial and will often sing loudly to defend their space."
        },
        2: {
            "name": "Periparus ater (Coal Tit)",
            "description": "The Coal Tit is a small songbird with a distinctive black cap. It's a forest bird but can also be found in gardens.",
            "habitat": "Woodland and gardens, especially coniferous forests.",
            "conservation_status": "Least Concern",
            "fun_fact": "Coal Tits are known for their ability to hide food in the bark of trees to retrieve later."
        },
        3: {
            "name": "Pica pica (Eurasian Magpie)",
            "description": "The Eurasian Magpie is a highly intelligent bird with striking black and white plumage and a long tail.",
            "habitat": "Open woodlands, farmland, and urban areas.",
            "conservation_status": "Least Concern",
            "fun_fact": "Magpies are known for their curiosity and intelligence, often solving complex puzzles and using tools."
        }
    }

    return species_info.get(species_id, None)

def app():
    st.title("Bird Species Detection for Conservation Research")
    st.subheader("Identify Erithacus rubecula, Periparus ater, and Pica pica in Uploaded Images")

    st.markdown("""
        This tool uses advanced computer vision techniques to detect and classify bird species, specifically Erithacus rubecula (European Robin), 
        Periparus ater (Coal Tit), and Pica pica (Eurasian Magpie). By identifying these species accurately, this project contributes to wildlife conservation efforts, 
        enabling researchers to track bird populations and study behavioral patterns over time.
        
        Our team evaluated two models, SSD and Faster R-CNN, and found that Faster R-CNN performed significantly better for bird species classification, achieving an accuracy of 92%.
        """)

    uploaded_file = st.file_uploader("Upload an Image (jpg, jpeg, or png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        PATH_TO_MODEL_DIR = r'saved_model\saved_model'
        PATH_TO_LABELS = 'label_map.pbtxt'

        detect_fn = load_model(PATH_TO_MODEL_DIR)

        category_index = load_label_map(PATH_TO_LABELS)

        image = Image.open(uploaded_file)
        image_np = load_image_into_numpy_array(image)

        result_image, detections = run_inference(detect_fn, image_np, category_index)

        st.image(result_image, caption="Detected Bird Species", use_container_width=True)

        detected_classes = detections['detection_classes'].astype(int)
        detection_scores = detections['detection_scores']

        for i, score in enumerate(detection_scores):
            if score > 0.75:
                species_id = detected_classes[i]
                if species_id in [1, 2, 3]:
                    species_info = display_species_info(species_id)
                    if species_info:
                        st.subheader(species_info["name"])
                        st.markdown(f"**Description:** {species_info['description']}")
                        st.markdown(f"**Habitat:** {species_info['habitat']}")
                        st.markdown(f"**Conservation Status:** {species_info['conservation_status']}")
                        st.markdown(f"**Fun Fact:** {species_info['fun_fact']}")

if __name__ == "__main__":
    app()
