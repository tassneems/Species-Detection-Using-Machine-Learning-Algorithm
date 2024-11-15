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

# Load the model based on the selected type
def load_model(model_type):
    """Load saved model and build the detection function based on model_type."""
    if model_type == 'Faster R-CNN':
        model_dir = r'saved_model_faster_rcnn\saved_model'
    else:
        model_dir = r'saved_model_ssd\saved_model'  # Example for SSD model

    print(f'Loading {model_type} model...', end='')
    start_time = time.time()
    detect_fn = tf.saved_model.load(model_dir)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done! Took {elapsed_time} seconds')
    return detect_fn

# Load the label map
def load_label_map(path_to_labels):
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    return category_index

    # Convert the image to a numpy array and ensure it has 3 channels (RGB)
def load_image_into_numpy_array(image):
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


# Run inference on the uploaded image
def run_inference(detect_fn, image_np, category_index, thresh=0.5):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = detect_fn(input_tensor)

    # Extract results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(int)

    image_np_with_detections = image_np.copy()

    # Visualize the detected boxes and labels
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

    # Return the info for the species
    return species_info.get(species_id, None)

def app():
    st.title("Bird Species Detection for Conservation Research")

    st.markdown("""
        This tool uses advanced computer vision techniques to detect and classify bird species, specifically Erithacus rubecula (European Robin), 
        Periparus ater (Coal Tit), and Pica pica (Eurasian Magpie). By identifying these species accurately, this project contributes to wildlife conservation efforts, 
        enabling researchers to track bird populations and study behavioral patterns over time.
    """)

    # Option for the user to select the model type
    model_type = st.radio("Select the Model Type", ('Faster R-CNN', 'SSD'))

    # Upload the image
    uploaded_file = st.file_uploader("Upload an Image (jpg, jpeg, or png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Load the selected model and label map
        PATH_TO_LABELS = 'label_map.pbtxt'  # Path to your label map file
        detect_fn = load_model(model_type)

        # Load the label map
        category_index = load_label_map(PATH_TO_LABELS)

        # Convert the uploaded image to PIL and then to numpy array
        image = Image.open(uploaded_file)
        image_np = load_image_into_numpy_array(image)

        # Run inference
        result_image, detections = run_inference(detect_fn, image_np, category_index)

        # Display the resulting image with bounding boxes and labels
        st.image(result_image, caption="Detected Bird Species", use_container_width=True)

        # Retrieve and display species information based on detection results
        detected_classes = detections['detection_classes'].astype(int)  # Get detected class IDs
        detection_scores = detections['detection_scores']  # Get detection scores

        # Filter classes that have detection scores above a threshold (e.g., 0.75)
        for i, score in enumerate(detection_scores):
            if score > 0.75:  # Check if the detection score is above threshold
                species_id = detected_classes[i]
                # Only display information for the 3 species we're interested in
                if species_id in [1, 2, 3]:
                    species_info = display_species_info(species_id)
                    if species_info:
                        st.markdown(f"### {species_info['name']}")
                        st.markdown(f"**Description:** {species_info['description']}")
                        st.markdown(f"**Habitat:** {species_info['habitat']}")
                        st.markdown(f"**Conservation Status:** {species_info['conservation_status']}")
                        st.markdown(f"**Fun Fact:** {species_info['fun_fact']}")

if __name__ == "__main__":
    app()
