# import cv2
# import numpy as np

# def compute_sift(image_path):
#     # Load the image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
#     # Create SIFT detector
#     sift = cv2.SIFT_create()
    
#     # Detect and compute keypoints and descriptors
#     keypoints, descriptors = sift.detectAndCompute(image, None)
    
#     return keypoints, descriptors

# def compute_similarity_score(image1_path, image2_path):
#     # Compute SIFT features for both images
#     keypoints1, descriptors1 = compute_sift(image1_path)
#     keypoints2, descriptors2 = compute_sift(image2_path)
    
#     # Create a Brute Force Matcher
#     bf = cv2.BFMatcher()
    
#     # Match descriptors
#     matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
    
#     # Calculate the similarity score
#     similarity_score = len(good_matches) / max(len(descriptors1), len(descriptors2))
    
#     return similarity_score

# # Example usage
# image1_path = 'lefthall.jpeg'
# image2_path = 'righthall.jpeg'

# score = compute_similarity_score(image1_path, image2_path)
# print(f"Similarity Score: {score}")


# --------------------------------------------------------------------------------------------

# import cv2
# import numpy as np

# def compute_sift(image):
#     # Convert to grayscale if the input is an image path
#     if isinstance(image, str):
#         image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
#     # Create SIFT detector
#     sift = cv2.SIFT_create()
    
#     # Detect and compute keypoints and descriptors
#     keypoints, descriptors = sift.detectAndCompute(image, None)
    
#     return keypoints, descriptors

# def compute_similarity_score(image1_keypoints, image1_descriptors, frame):
#     # Compute SIFT features for the frame
#     keypoints, descriptors = compute_sift(frame)
    
#     # Create a Brute Force Matcher
#     bf = cv2.BFMatcher()
    
#     # Match descriptors
#     matches = bf.knnMatch(image1_descriptors, descriptors, k=2)
    
#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
    
#     # Calculate the similarity score
#     similarity_score = len(good_matches) / max(len(image1_descriptors), len(descriptors))
    
#     return similarity_score

# # Example usage
# image1_path = '10.jpg'
# video_path = 'cctv.mp4'

# # Compute SIFT features for the reference image
# image1_keypoints, image1_descriptors = compute_sift(image1_path)

# # Open the video capture
# cap = cv2.VideoCapture(video_path)

# # Process every 4th frame
# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     frame_count += 1
    
#     # Process every 4th frame
#     if frame_count % 4 == 0:
#         # Convert the frame to grayscale
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Compute similarity score
#         score = compute_similarity_score(image1_keypoints, image1_descriptors, frame_gray)
        
#         print(f"Similarity Score for Frame {frame_count}: {score}")

# # Release the video capture
# cap.release()

#-------------------------------------------------------------------


# import cv2
# import numpy as np
# from playsound import playsound

# def alert_sound():
#     playsound(r'D:\Desktop\Similarty\alert.mp3')
#     return

# def compute_sift(image):
#     # Convert to grayscale if the input is an image path
#     if isinstance(image, str):
#         image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
#     # Create SIFT detector
#     sift = cv2.SIFT_create()
    
#     # Detect and compute keypoints and descriptors
#     keypoints, descriptors = sift.detectAndCompute(image, None)
    
#     return keypoints, descriptors

# def compute_similarity_score(image1_keypoints, image1_descriptors, frame):
#     # Compute SIFT features for the frame
#     keypoints, descriptors = compute_sift(frame)
    
#     # Create a Brute Force Matcher
#     bf = cv2.BFMatcher()
    
#     # Match descriptors
#     matches = bf.knnMatch(image1_descriptors, descriptors, k=2)
    
#     # Apply ratio test
#     good_matches = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:
#             good_matches.append(m)
    
#     # Calculate the similarity score
#     similarity_score = len(good_matches) / max(len(image1_descriptors), len(descriptors))
    
#     return similarity_score

# # Example usage
# image1_path = r'righthall.jpeg'
# video_path = r'lefthl.mp4'

# # Compute SIFT features for the reference image
# image1_keypoints, image1_descriptors = compute_sift(image1_path)

# # Open the video capture
# cap = cv2.VideoCapture(video_path)

# # Get video frame dimensions
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Set the window size for display
# window_width = 600

# # Calculate window_height if frame dimensions are non-zero, otherwise set a default value
# window_height = int(frame_height * (window_width / frame_width)) if frame_width != 0 and frame_height != 0 else 500

# # Process every 4th frame
# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     frame_count += 1
    
#     # Process every 4th frame
#     if frame_count % 4 == 0:
#         # Resize the reference image to match the height of the video frame
#         image1_resized = cv2.resize(cv2.imread(image1_path), (frame_width, frame_height))
        
#         # Convert the frame to grayscale
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Compute similarity score
#         score = compute_similarity_score(image1_keypoints, image1_descriptors, frame_gray)
        
#         # Display image1 and the current frame
#         concatenated_img = np.concatenate((image1_resized, frame), axis=1)
        
#         # Resize the concatenated image to fit within the screen dimensions
#         if concatenated_img.shape[1] > window_width:
#             scale_factor = window_width / concatenated_img.shape[1]
#             concatenated_img = cv2.resize(concatenated_img, (window_width, int(concatenated_img.shape[0] * scale_factor)))
        
#         # Display similarity message
#         if score > 0.1:
#             message = f"Similarity Score for Frame {frame_count}: {score} - Similar Images"
#         else:
#             message = f"Similarity Score for Frame {frame_count}: {score} - Not Similar"
#             alert_sound()

#         print(message)
        
#         # Draw the text on the image
#         cv2.putText(concatenated_img, message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
#         # Display the image with text
#         cv2.imshow('Similarity Comparison', concatenated_img)
        
#         # Wait for a key press and break if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the video capture
# cap.release()
# cv2.destroyAllWindows()


#------------------------------------------------------------------------

import cv2
import numpy as np
import pygame

def alert_sound():
    pygame.init()  # Initialize the video system
    pygame.mixer.init()
    pygame.mixer.music.load(r'D:\Desktop\Similarty\alert.mp3')
    pygame.mixer.music.play()
    pygame.event.wait()
    return

def compute_sift(image):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors

def compute_similarity_score(image1_keypoints, image1_descriptors, frame):
    keypoints, descriptors = compute_sift(frame)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(image1_descriptors, descriptors, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    similarity_score = len(good_matches) / max(len(image1_descriptors), len(descriptors))
    
    return similarity_score

# Example usage
image1_path = r'righthall.jpeg'
video_path = r'lefthl.mp4'

image1_keypoints, image1_descriptors = compute_sift(image1_path)

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

window_width = 600
window_height = int(frame_height * (window_width / frame_width)) if frame_width != 0 and frame_height != 0 else 500

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 4 == 0:
        image1_resized = cv2.resize(cv2.imread(image1_path), (frame_width, frame_height))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = compute_similarity_score(image1_keypoints, image1_descriptors, frame_gray)
        
        concatenated_img = np.concatenate((image1_resized, frame), axis=1)
        
        if concatenated_img.shape[1] > window_width:
            scale_factor = window_width / concatenated_img.shape[1]
            concatenated_img = cv2.resize(concatenated_img, (window_width, int(concatenated_img.shape[0] * scale_factor)))
        
        if score > 0.1:
            message = f"Similarity Score for Frame {frame_count}: {score} - Similar Images"
        else:
            message = f"Similarity Score for Frame {frame_count}: {score} - Not Similar"
            alert_sound()

        print(message)
        
        cv2.putText(concatenated_img, message, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('Similarity Comparison', concatenated_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
