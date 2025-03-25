"""
This script downloads all videos in the dataset with label_state 16, 20, or 32.
(Non-smoke)
"""

import sys
import urllib.request
import os
import json


def is_file_here(file_path):
    """
    Check if a file exists.

    Parameters
    ----------
    file_path : str
        The file path that we want to check.

    Returns
    -------
    bool
        If the file exists (True) or not (False).
    """
    return os.path.isfile(file_path)


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


def get_video_url(v):
    """
    Get the video URL.

    Parameters
    ----------
    v : dict
        The dictionary with keys and values in the video dataset JSON file.

    Returns
    -------
    str
        The full URL of the video.
    """
    camera_names = ["hoogovens", "kooksfabriek_1", "kooksfabriek_2"]
    return v["url_root"] + camera_names[v["camera_id"]] + "/" +  v["url_part"] + "/" + v["file_name"] + ".mp4"


def main(argv):
    # Specify the path to the JSON file
    json_file_path = "metadata_ijmond_jan_22_2024.json"

    # Specify the path that we want to store the videos and create it
    download_path = "videos/"
    check_and_create_dir(download_path)

    # Open the file and load its contents into a dictionary
    with open(json_file_path, "r") as json_file:
        data_dict = json.load(json_file)

    # Define allowed label_states
    allowed_labels = {16, 20, 32}


    # Start to download files
    problem_videos = []
    for v in data_dict:
        # Filter the dictionary based on allowed label_states
        if v["label_state"] not in allowed_labels:
            continue
        video_url = get_video_url(v)
        file_name = v["file_name"]
        file_path = download_path + file_name + ".mp4"
        if not is_file_here(file_path):
            print("Download video", file_name)
            try:
                urllib.request.urlretrieve(video_url, file_path)
            except Exception:
                print("\tError downloading video", file_name)
                problem_videos.append(file_name)

    # Print errors
    if len(problem_videos) > 0:
        print("The following videos were not downloaded due to errors:")
        for v in problem_videos:
            print("\tv\n")
    print("DONE")


if __name__ == "__main__":
    main(sys.argv)