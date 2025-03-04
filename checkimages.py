import os
from collections import defaultdict

def check_image_angles(folder):
    image_dict = defaultdict(list)

    # List all image files
    for filename in os.listdir(folder):
        if filename.endswith(( '.png')):  # Adjust extensions if needed
            parts = filename.split('-')
            if len(parts) == 3:  # Ensure the filename follows expected format
                base_id = f"{parts[0]}-{parts[1]}"
                image_dict[base_id].append(filename)

    # Print images that do not have exactly 2 angles
    for base_id, images in image_dict.items():
        if len(images) > 3:
            print(f"{base_id}: {len(images)} angles")
            for img in images:
                print(f"  - {img}")

# Example usage
folder_path = "data/iu_xray/images"  # Change this to your actual folder path
check_image_angles(folder_path)
