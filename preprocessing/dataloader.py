import random
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import cv2
import logging


logger = logging.basicConfig(filename="rppg.app", filemode='w', level=logging.DEBUG)

def sample_images(rgb_faces, thermal_faces, labels, target_length=5376):
    # Get the number of images in each modality
    num_rgb = len(rgb_faces)
    num_thermal = len(thermal_faces)
    
    # Determine the minimum number of images
    min_length = min(num_rgb, num_thermal, len(labels))
    
    # If the minimum length is greater than the target length, sample down to target length
    if min_length > target_length:
        sampled_indices = sorted(random.sample(range(min_length), target_length))
    else:
        # If already less than or equal to target length, use all images up to min_length
        sampled_indices = list(range(min_length))
    
    # Sample the images and labels based on the sampled indices
    rgb_faces_sampled = [rgb_faces[i] for i in sampled_indices]
    thermal_faces_sampled = [thermal_faces[i] for i in sampled_indices]
    labels_sampled = [labels[i] for i in sampled_indices]
    
    return rgb_faces_sampled, thermal_faces_sampled, labels_sampled



def load_iBVP_dataset(dataset_path, target_length=5376):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    all_rgb_faces = []
    all_thermal_faces = []
    all_labels = []

    thermal_img_width = 640
    thermal_img_height = 512

    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
        
        rgb_folder = os.path.join(person_path, f"{person_folder}_rgb")
        thermal_folder = os.path.join(person_path, f"{person_folder}_t")
        csv_file = os.path.join(person_path, f"{person_folder}_bvp.csv")

        # Load the BVP data
        bvp_data = pd.read_csv(csv_file)
        labels = bvp_data.iloc[:, 0].values  # Extract the first column

        rgb_faces = []
        thermal_faces = []
        
        for img_name in sorted(os.listdir(rgb_folder)):
            img_path = os.path.join(rgb_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform face detection
            results = face_detection.process(img_rgb)

            if results.detections:
                # Get the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face = img_rgb[y:y+h, x:x+w]

                # Resize the face to 64x64
                face_resized = cv2.resize(face, (64, 64))
                rgb_faces.append(face_resized)
            else:
                # If no face detected, use a placeholder (black image)
                rgb_faces.append(np.zeros((64, 64, 3), dtype=np.uint8))

        for img_name in sorted(os.listdir(thermal_folder)):
            img_path = os.path.join(thermal_folder, img_name)
            
            # Read and process the thermal .raw file
            thermal_matrix = np.fromfile(img_path, dtype=np.uint16, count=thermal_img_width * thermal_img_height).reshape(thermal_img_height, thermal_img_width)
            thermal_matrix = thermal_matrix.astype(np.float32)
            thermal_matrix = (thermal_matrix * 0.04) - 273.15  # Convert to Celsius

            # Normalize the thermal matrix for visualization purposes (optional)
            thermal_matrix_normalized = cv2.normalize(thermal_matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Perform face detection on normalized thermal image
            thermal_img_colormap = cv2.applyColorMap(thermal_matrix_normalized, cv2.COLORMAP_JET)
            thermal_img_rgb = cv2.cvtColor(thermal_img_colormap, cv2.COLOR_BGR2RGB)
            results = face_detection.process(thermal_img_rgb)

            if results.detections:
                # Get the first detected face
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = thermal_img_rgb.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face_temp = thermal_matrix[y:y+h, x:x+w]

                # Resize the face to 64x64
                face_temp_resized = cv2.resize(face_temp, (64, 64))
                thermal_faces.append(face_temp_resized)
            else:
                # If no face detected, use a placeholder (zeros matrix)
                thermal_faces.append(np.zeros((64, 64), dtype=np.float32))

        # Ensure consistent length by sampling
        rgb_faces_sampled, thermal_faces_sampled, labels_sampled = sample_images(rgb_faces, thermal_faces, labels, target_length=target_length)

        all_rgb_faces.append(rgb_faces_sampled)
        all_thermal_faces.append(thermal_faces_sampled)
        all_labels.append(labels_sampled)

    # Convert lists to NumPy arrays
    all_rgb_faces_np = np.array(all_rgb_faces)
    all_thermal_faces_np = np.array(all_thermal_faces)
    all_labels_np = np.array(all_labels)

    logging.DEBUG("collected datasets, preparing for preprocessing")

    return all_rgb_faces_np, all_thermal_faces_np, all_labels_np


