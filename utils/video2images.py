import cv2
import os 

def extract_frames(path, filename, start_frame, frame_interval):
    
    output_folder = os.path.join(path, 'imgs', os.path.splitext(filename)[0])
    input_video = os.path.join(path, 'videos', filename)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    else:
        print(f"Folder '{output_folder}' already exists.")
    
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    name_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            # Save the frame as an image
            frame_filename = f"{output_folder}/frame_{name_count:04d}.jpg"
            name_count += 1
            cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    # Release the video file
    cap.release()
    
    print(f"{name_count} frames extracted and saved to {output_folder}")


if __name__ == '__main__':
    # Specify the input video file, output folder, starting frame, and frame interval
    path = "datasets/eth_ucy"
    start_frame = 0
    frame_interval = 10

    # Call the function to extract frames
    video_folder = os.path.join(path, "videos")
    for filename in os.listdir(video_folder):
        if not filename.endswith(".avi"):
            continue
        
        video_path = os.path.join(video_folder, filename)
        extract_frames(path, filename, start_frame, frame_interval)
        