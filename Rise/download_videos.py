import json
import os
import requests
from urllib.parse import urljoin

# Configure these parameters
JSON_FILE = 'metadata_02242020.json'  # Path to your JSON file
TARGET_LABELS = {32,16}  # Labels to download
OUTPUT_DIR = 'Strong_negative'  # Directory to save videos
MAX_RETRIES = 3  # Download retry attempts


def download_videos():
    # Load JSON data
    with open(JSON_FILE) as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create session with retries
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=MAX_RETRIES)
    session.mount('https://', adapter)

    # Process each entry
    for entry in data:
        if entry.get('label_state') in TARGET_LABELS:
            try:
                # Construct URL
                video_url = urljoin(entry['url_root'], entry['url_part'])

                # Create filename
                filename = f"{entry['file_name']}.mp4"
                save_path = os.path.join(OUTPUT_DIR, filename)

                # Skip if already exists
                if os.path.exists(save_path):
                    print(f"Skipping existing file: {filename}")
                    continue

                # Download video
                print(f"Downloading {video_url}...")
                response = session.get(video_url, stream=True, timeout=30)
                response.raise_for_status()

                # Save file
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"Successfully saved: {filename}")

            except Exception as e:
                print(f"Failed to download {entry['file_name']}: {str(e)}")


if __name__ == '__main__':
    download_videos()