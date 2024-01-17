# import cv2
# from roboflow import Roboflow
# import datetime

# rf = Roboflow(api_key="gfENJ7NUVSKxc9RmM4Uw")
# project = rf.workspace().project("violence-weapon-detection")
# model = project.version(1).model

# # Function to predict on a frame and return results
# def predict_on_frame(frame):
#     # Convert the frame to the required format (BGR to RGB for Roboflow)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Make predictions on the frame
#     result = model.predict(frame_rgb, confidence=30, overlap=50).json()
    
#     # predictions = result.get('predictions', [])
#     predictions = result.get('predictions', [])
    
#     for prediction in predictions:
#         confidence = prediction.get('confidence', None)
#         predicted_class = prediction.get('class', None)
        
#         if predicted_class != 'NonViolence':
#             print(f"Class: {predicted_class}, Confidence: {confidence}, Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
#         elif predicted_class == "NonViolence":
#             print("NONVIOLENCE")
#         else:
#             print("Nothing detected")

#     # Show the frame without bounding boxes
#     cv2.imshow("Video", frame)
#     cv2.waitKey(1)

# # Function to predict on frames extracted from a video with a specific FPS
# def predict_on_video(video_path, target_fps):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get the frames per second (FPS) of the video
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Calculate the frame interval to achieve the target FPS
#     frame_interval = int(round(fps / target_fps))

#     # Read frames from the video
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()

#         # Break the loop if no more frames are available
#         if not ret:
#             break

#         # Only process frames at the specified interval
#         if frame_count % frame_interval == 0:
#             # Make predictions on the frame
#             predict_on_frame(frame)

#         frame_count += 1

#     # Release the video capture object
#     cap.release()

# # Example usage for video input with 4 frames per second
# video_path = "video.mp4"
# target_fps = 4
# predict_on_video(video_path, target_fps)

# # Destroy OpenCV window
# cv2.destroyAllWindows()

# import cv2
# from roboflow import Roboflow
# import datetime

# rf = Roboflow(api_key="gfENJ7NUVSKxc9RmM4Uw")
# project = rf.workspace().project("violence-weapon-detection")
# model = project.version(1).model

# # Function to predict on a frame and return results
# def predict_on_frame(frame):
#     # Convert the frame to the required format (BGR to RGB for Roboflow)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Make predictions on the frame
#     result = model.predict(frame_rgb, confidence=30, overlap=50).json()
    
#     # predictions = result.get('predictions', [])
#     predictions = result.get('predictions', [])
    
#     for prediction in predictions:
#         confidence = prediction.get('confidence', None)
#         predicted_class = prediction.get('class', None)
        
#         if predicted_class != 'NonViolence':
#             print(f"Class: {predicted_class}, Confidence: {confidence}, Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
#         elif predicted_class == "NonViolence":
#             print("NONVIOLENCE")
#         else:
#             print("Nothing detected")

#         # Draw bounding box on the frame
#         x, y, width, height = int(prediction.get('x')), int(prediction.get('y')), int(prediction.get('width')), int(prediction.get('height'))
#         cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
#         cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (x, y - 10), 0, 0.5, (0, 255, 0), 2)

#     # Show the frame with results
#     cv2.imshow("Predictions", frame)
#     cv2.waitKey(1)

# # Function to predict on frames extracted from a video with a specific FPS
# def predict_on_video(video_path, target_fps):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get the frames per second (FPS) of the video
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Calculate the frame interval to achieve the target FPS
#     frame_interval = int(round(fps / target_fps))

#     # Read frames from the video
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()

#         # Break the loop if no more frames are available
#         if not ret:
#             break

#         # Only process frames at the specified interval
#         if frame_count % frame_interval == 0:
#             # Make predictions on the frame
#             predict_on_frame(frame)

#         frame_count += 1

#     # Release the video capture object
#     cap.release()

# # Example usage for video input with 4 frames per second
# video_path = "video.mp4"
# target_fps = 4
# predict_on_video(video_path, target_fps)

# # Destroy OpenCV window
# cv2.destroyAllWindows()
#////////////////////////////////////////////////////////////////////////////







# import cv2
# from roboflow import Roboflow
# import datetime
# from playsound import playsound

# rf = Roboflow(api_key="gfENJ7NUVSKxc9RmM4Uw")
# project = rf.workspace().project("violence-weapon-detection")
# model = project.version(1).model

# def alert_sound():
#     playsound(r'D:\Desktop\violese\alert.mp3')
#     return

# # Function to predict on a frame and return results
# def predict_on_frame(frame):
#     # Convert the frame to the required format (BGR to RGB for Roboflow)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Make predictions on the frame
#     result = model.predict(frame_rgb, confidence=40, overlap=50).json()
    
#     # predictions = result.get('predictions', [])
#     predictions = result.get('predictions', [])
    
#     for prediction in predictions:
#         confidence = prediction.get('confidence', None)
#         predicted_class = prediction.get('class', None)
        
#         if predicted_class != 'NonViolence':
#             alert_sound()
#         #     print(f"Class: {predicted_class}, Confidence: {confidence}, Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
#         # elif predicted_class == "NonViolence":
#         #     print("NONVIOLENCE")
#         # else:
#         #     print("Nothing detected")

#     # Display the text above the frame window
#         cv2.putText(frame, f"Class: {predicted_class}, Confidence: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#         # Show the frame with results
#         cv2.imshow("Predictions", frame)
#         cv2.waitKey(1)

# # Function to predict on frames extracted from a video with a specific FPS
# def predict_on_video(video_path, target_fps):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Check if the video opened successfully
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get the frames per second (FPS) of the video
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # Calculate the frame interval to achieve the target FPS
#     frame_interval = int(round(fps / target_fps))

#     # Read frames from the video
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()

#         # Break the loop if no more frames are available
#         if not ret:
#             break

#         # Only process frames at the specified interval
#         if frame_count % frame_interval == 0:
#             # Make predictions on the frame
#             predict_on_frame(frame)

#         frame_count += 1

#     # Release the video capture object
#     cap.release()

# # Example usage for video input with 4 frames per second
# video_path = "crime.mp4"
# target_fps = 2
# predict_on_video(video_path, target_fps)

# # Destroy OpenCV window
# cv2.destroyAllWindows()




import cv2
from roboflow import Roboflow
import datetime
from playsound import playsound

rf = Roboflow(api_key="gfENJ7NUVSKxc9RmM4Uw")
project = rf.workspace().project("violence-weapon-detection")
model = project.version(1).model

def alert_sound():
    playsound(r'D:\Desktop\violese\alert.mp3')
    return

def predict_on_frames(frames):
    for frame in frames:
        # Convert the frame to the required format (BGR to RGB for Roboflow)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make predictions on the frame
        result = model.predict(frame_rgb, confidence=40, overlap=50).json()
        
        predictions = result.get('predictions', 0)

        if not predictions:
            predicted_class = 'nothing detected'
            confidence = 1
        
        for prediction in predictions:
            confidence = prediction.get('confidence', None)
            predicted_class = prediction.get('class', None)
            
            if predicted_class != 'NonViolence':
                alert_sound()

            # Display the text above the frame window
        cv2.putText(frame, f"Class: {predicted_class}, Confidence: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show the frame with results
        cv2.imshow("Predictions", frame)
        cv2.waitKey(1)

# Function to predict on frames extracted from a video with a specific FPS
def predict_on_video(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps / target_fps))

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    # Call the function to predict on frames
    predict_on_frames(frames)

    cv2.destroyAllWindows()

# Example usage for video input with 2 frames per second
video_path = "hand.mp4"
target_fps = 1
predict_on_video(video_path, target_fps)
