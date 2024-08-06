import os
from ultralytics import YOLO
import cv2
import moviepy.editor as mp 
import speech_recognition as sr 
current_dir = os.path.dirname(os.path.abspath(__file__))

def video_2_text(media_path):
    # Load the video 
    vdtxt = mp.VideoFileClip(media_path) 

    basename = media_path.rsplit('.', 1)[0]
    wav_file = basename + ".wav"
    # Extract the audio from the video 
    audio_file = vdtxt.audio 
    audio_file.write_audiofile(wav_file) 

    # Initialize recognizer 
    r = sr.Recognizer() 

    # Load the audio file 
    with sr.AudioFile(wav_file) as source: 
        data = r.record(source) 

    try:
        text = r.recognize_google(data)
        print("\nThe resultant text from video is: \n", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

#-----------------------------------------------------------------------------------

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read the first frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        return

    # Release the video capture object
    cap.release()

    # Resize the frame to 640x640 while maintaining aspect ratio
    frame_resized = resize_and_pad(frame, (640, 640))

    # Save the first frame as an image
    first_frame_path = "first_frame.jpg"
    cv2.imwrite(first_frame_path, frame_resized)

    # Process the first frame as an image
    return process_image(first_frame_path)

def resize_and_pad(image, size):
    h, w, _ = image.shape
    sh, sw = size

    # Resize while keeping aspect ratio
    aspect = w / h
    if aspect > 1:
        new_w = sw
        new_h = int(sw / aspect)
    else:
        new_h = sh
        new_w = int(sh * aspect)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new image with the target size and place the resized image in the center
    new_image = cv2.copyMakeBorder(resized_image, 
                                   (sh - new_h) // 2, (sh - new_h + 1) // 2, 
                                   (sw - new_w) // 2, (sw - new_w + 1) // 2, 
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return new_image

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file {image_path}")
        return

    # Load the custom YOLO model
    model_path = os.path.join(current_dir,'runs', 'detect', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)

    # Set the detection threshold
    threshold = 0.4  # Increased threshold to reduce false positives

    # Dictionary to store detected machine names and their counts
    detected_machine_counts = {}

    # Perform detection on the image
    results = model(image)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            detected_name = results.names[int(class_id)].upper()
            # Apply size filtering
            if (x2 - x1) < 50 or (y2 - y1) < 50:  # Example size filtering
                continue
            if detected_name in detected_machine_counts:
                detected_machine_counts[detected_name] += 1
            else:
                detected_machine_counts[detected_name] = 1
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, detected_name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image with detections
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find the most common detected machine name
    if detected_machine_counts:
        most_common_name = max(detected_machine_counts, key=detected_machine_counts.get)
        print(f"The most frequently detected machine name is: {most_common_name}")
        if most_common_name == "FRAISEUSE":
            most_common_name = "MILLING MACHINE"
        elif most_common_name == "PERCEUSE":
            most_common_name = "DRILLING MACHINE"
        elif most_common_name == "TOUR PARALLELE":
            most_common_name = "LATHE MACHINE"
        return most_common_name
    else:
        print("No machines were detected.")

# Define the path to your media file
def yolo_media_upload(media):
    # Check if the file is a video or an image
    if media.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more video formats if needed
        return video_2_text(media),process_video(media)
    elif media.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # Add more image formats if needed
        process_image(media)
    else:
        print(f"Unsupported file format: {media}")
