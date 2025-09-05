import os
import shutil

yolo_dataset_dir = r"C:\Users\Admin\Desktop\output_yolo_dataset\converted"


output_dir = r"C:\Users\Admin\Desktop\output_yolo_dataset\remapped"

class_map = {
    0: 2,   # Aluminium foil -> metal
    1: 2,   # Battery -> metal
    2: 2,   # Aluminium blister pack -> metal
    3: 3,   # Carded blister pack -> paper
    4: 4,   # Other plastic bottle -> plastic
    5: 4,   # Clear plastic bottle -> plastic
    6: 1,   # Glass bottle -> glass
    7: 4,   # Plastic bottle cap -> plastic
    8: 2,   # Metal bottle cap -> metal
    9: 1,   # Broken glass -> glass
    10: 2,  # Food Can -> metal
    11: 2,  # Aerosol -> metal
    12: 2,  # Drink can -> metal
    13: 3,  # Toilet tube -> paper
    14: 0,  # Other carton -> cardboard
    15: 0,  # Egg carton -> cardboard
    16: 0,  # Drink carton -> cardboard
    17: 0,  # Corrugated carton -> cardboard
    18: 0,  # Meal carton -> cardboard
    19: 0,  # Pizza box -> cardboard
    20: 3,  # Paper cup -> paper
    21: 4,  # Disposable plastic cup -> plastic
    22: 3,  # Foam cup -> paper
    23: 1,  # Glass cup -> glass
    24: 4,  # Other plastic cup -> plastic
    25: 6,  # Food waste -> organic_trash
    26: 1,  # Glass jar -> glass
    27: 4,  # Plastic lid -> plastic
    28: 2,  # Metal lid -> metal
    29: 4,  # Other plastic -> plastic
    30: 3,  # Magazine paper -> paper
    31: 3,  # Tissues -> paper
    32: 3,  # Wrapping paper -> paper
    33: 3,  # Normal paper -> paper
    34: 3,  # Paper bag -> paper
    35: 3,  # Plastified paper bag -> paper
    36: 4,  # Plastic film -> plastic
    37: 5,  # Garbage bag -> trash
    38: 4,  # Other plastic wrapper -> plastic
    39: 4,  # Single-use carrier bag -> plastic
    40: 4,  # Polypropylene bag -> plastic
    41: 4,  # Crisp packet -> plastic
    42: 5,  # Spread tub -> general_litter
    43: 5,  # Tupperware -> general_litter
    44: 7,  # Disposable food container -> styrofoam_trash
    45: 7,  # Foam food container -> styrofoam_trash
    46: 4,  # Other plastic container -> plastic
    47: 4,  # Plastic gloves -> plastic
    48: 4,  # Plastic utensils -> plastic
    49: 2,  # Pop tab -> metal
    50: 5,  # Rope & strings -> general_litter
    51: 2,  # Scrap metal -> metal
    52: 5,  # Shoe -> general_litter
    53: 5,  # Squeezable tube -> general_litter
    54: 4,  # Plastic straw -> plastic
    55: 3,  # Paper straw -> paper
    56: 7,  # Styrofoam piece -> styrofoam_trash
    57: 5,  # Unlabeled litter -> general_litter
    58: 5   # Cigarette -> general_litter
}

# Final new class names
class_names = [
    "cardboard", "glass", "metal", "paper", "plastic", "general_litter", "organic_trash", "styrofoam_trash"
]

# =========================
# PROCESS DATA
# =========================

for subset in ["train", "test"]:
    src_images = os.path.join(yolo_dataset_dir, "images", subset)
    src_labels = os.path.join(yolo_dataset_dir, "labels", subset)

    dst_images = os.path.join(output_dir, "images", subset)
    dst_labels = os.path.join(output_dir, "labels", subset)
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    for label_file in os.listdir(src_labels):
        if not label_file.endswith(".txt"):
            continue
        src_label_path = os.path.join(src_labels, label_file)
        dst_label_path = os.path.join(dst_labels, label_file)

        new_lines = []
        with open(src_label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                # Each object has 5 parts: class_id x y w h
                for i in range(0, len(parts), 5):
                    old_id = int(parts[i])
                    if old_id in class_map:
                        new_id = class_map[old_id]
                        obj_line = [str(new_id)] + parts[i+1:i+5]
                        new_lines.append(" ".join(obj_line))
        if not new_lines:
            continue  # skip empty labels

        # Copy image
        src_image_path = os.path.join(src_images, label_file.replace(".txt", ".jpg"))
        if not os.path.exists(src_image_path):
            src_image_path = os.path.join(src_images, label_file.replace(".txt", ".png"))
        shutil.copy(src_image_path, os.path.join(dst_images, os.path.basename(src_image_path)))

        # Write new label
        with open(dst_label_path, "w") as f:
            f.write("\n".join(new_lines))

# =========================
# CREATE data.yaml
# =========================

import yaml

yaml_dict = {
    "train": os.path.join(output_dir, "images/train"),
    "val": os.path.join(output_dir, "images/val"),
    "nc": len(class_names),
    "names": class_names
}

with open(os.path.join(output_dir, "data.yaml"), "w") as f:
    yaml.dump(yaml_dict, f)

print("✅ Remapped YOLO dataset ready at:", output_dir)
