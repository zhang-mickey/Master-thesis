import cv2
import os

def extract_frames(video_path, frame_paths, frame_indices=None):
    """
    Extract multiple frames from a video and save them as images.
    
    Parameters:
    -----------
    video_path : str
        Path to the video file
    frame_paths : list
        List of paths where to save the extracted frames
    frame_indices : list, optional
        List of frame indices to extract. If None, extracts first and middle frames.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return False
    
    # Determine which frames to extract
    if frame_indices is None:
        # Extract first frame and middle frame
        frame_indices = [0, total_frames // 2]
    
    success = True
    extracted_frames = []
    
    for idx, frame_idx in enumerate(frame_indices):
        # Set position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        
        if success:
            cv2.imwrite(frame_paths[idx], frame)
            extracted_frames.append(frame)
        else:
            break
    
    cap.release()
    return len(extracted_frames) == len(frame_paths)

def check_and_create_dir(dir_path):
    """
    Check and create a directory if it does not exist.

    Parameters
    ----------
    dir_path : str
        The dictionary path that we want to create.
    """
    if dir_path is None: return
    dir_name = os.path.dirname(dir_path)
    if dir_name != "" and not os.path.exists(dir_name):
        try: # This is used to prevent race conditions during parallel computing
            os.makedirs(dir_name)
        except Exception as ex:
            print(ex)
            
def main():
    # Path configurations
    video_dir = "videos/"  # Directory containing downloaded videos
    frames_dir = "frames/"  # Directory to save extracted frames

    # Create frames directory if it doesn't exist
    check_and_create_dir(frames_dir)

    # List all video files in the videos directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        base_filename = os.path.splitext(video_file)[0]
        
        # Define paths for two frames
        frame1_path = os.path.join(frames_dir, f"{base_filename}_frame1.jpg")
        frame2_path = os.path.join(frames_dir, f"{base_filename}_frame2.jpg")
        frame_paths = [frame1_path, frame2_path]
        
        # Skip if both frames already exist
        if os.path.exists(frame1_path) and os.path.exists(frame2_path):
            print(f"Frames already exist for: {video_file}")
            continue

        # Extract frames
        print(f"Extracting frames from: {video_file}")
        success = extract_frames(video_path, frame_paths)
        if not success:
            print(f"\tFailed to extract frames from {video_file}")

    print("DONE")


if __name__ == "__main__":
    main()