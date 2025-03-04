import os
import shutil

def process_images(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_groups = {}

    # Group images by their base name (everything except last field)
    for image in images:
        parts = image.rsplit('-', 1)  # Split only on the last '-'
        if len(parts) != 2:
            continue  # Skip invalid filenames
        
        base_name = parts[0]  # Everything before last '-'
        angle_id = parts[1].split('.')[0]  # Extract angle ID without extension

        if base_name not in image_groups:
            image_groups[base_name] = []
        image_groups[base_name].append((image, angle_id))

    # Process images with more than 2 angles
    for base_name, files in image_groups.items():
        if len(files) <= 2:
            continue  # Skip if there are not more than 2 angles

        files.sort(key=lambda x: x[1])  # Sort by angle ID
        new_folder = os.path.join(folder_path, base_name)
        os.makedirs(new_folder, exist_ok=True)

        # Rename the first 2 angles as "0.png" and "1.png"
        for i, (old_name, _) in enumerate(files[:2]):
            old_path = os.path.join(folder_path, old_name)
            new_name = f"{i}.png"
            new_path = os.path.join(new_folder, new_name)

            shutil.move(old_path, new_path)
            print(f"Renamed {old_name} -> {new_path}")

        # Delete remaining images
        for old_name, _ in files[2:]:
            old_path = os.path.join(folder_path, old_name)
            if os.path.exists(old_path):
                os.remove(old_path)
                print(f"Deleted {old_name}")

if __name__ == "__main__":
    folder_path = input("Enter folder path: ").strip()
    process_images(folder_path)
