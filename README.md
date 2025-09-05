# Smart Trash Detector 🗑️

AI-powered trash detection and recycling guidance using **YOLO** and **GPT-OSS**.

---

## Demo

- Upload an image of trash and get detected items highlighted.
- Receive short, clear recycling instructions via GPT-OSS.

---

## Features

- **Object Detection:** YOLOv8 model trained on custom trash dataset.
- **Instructions Generation:** GPT-OSS provides household recycling guidance.
- **Upload Image Tab:** Annotates and explains detected trash.
- **Webcam Tab:** Live detection (note: frame rate may vary).

---

## Dataset

We used the [TACO Dataset](https://huggingface.co/datasets/Zesky665/TACO) (Trash Annotations in Context).  

### Conversion to YOLO format

To train YOLO, annotations were converted from COCO to YOLO format. You can do this with a Python script:

```bash
python coco_to_yolo.py --coco-dir /path/to/TACO --output-dir /path/to/yolo --class-map classes.yaml
```
---
## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/smart-trash-detector.git
cd smart-trash-detector
pip install -r requirements.txt
```

## requirements.txt includes:

- ultralytics

- gradio

- huggingface-hub

- numpy

- Pillow

## Running the App

Make sure you have your YOLO model weights (`best.pt`) in the `models/` folder.

```bash
python app.py
```

- Upload an image in the "Upload Image" tab to detect trash and get GPT-OSS guidance.

- **Optional**: Use the "Webcam" tab for live detection.

## Training YOLO (Optional)

If you want to retrain YOLO on your own data:

```bash
python trainingYolo.py --data yolo_dataset.yaml --weights yolov8n.pt --epochs 50
```

- Replace yolo_dataset.yaml with your dataset config.

- The resulting model weights can be saved as best.pt.

## Notes

- GPT-OSS requires a Hugging Face API token set as HF_API_TOKEN.

- Live webcam detection may be slower and sometimes misclassifies objects.

- For the hackathon, you can include best.pt in the repo or provide a download link if the file is large.

- Make sure to include any data processing scripts (coco_to_yolo.py) if you want your repo fully reproducible.