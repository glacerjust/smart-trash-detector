import gradio as gr
from ultralytics import YOLO
import os
from huggingface_hub import InferenceClient
from collections import Counter

HF_TOKEN = os.environ.get("HF_API_TOKEN")
client = InferenceClient(provider="together", api_key=HF_TOKEN)

# Load your YOLO model
yolo = YOLO("models/best.pt")


def call_gptoss(prompt):
    if not HF_TOKEN:
        return "HF_API_TOKEN not set!"
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT-OSS: {e}"

def detections_to_prompt(detections):
    if not detections:
        return 'No trash detected'
    counts = Counter(detections)
    items = ", ".join([f"{v}× {k}" for k, v in counts.items()]) # count x items and join them 
    return f"The trash detector found: {items}. How should I dispose of these items? Keep it short." # return the prompt


def detect_and_explain(image):
    results = yolo.predict(image)
    annotated = results[0].plot()
    detections = [yolo.names[int(b.cls)] for b in results[0].boxes] if hasattr(results[0], "boxes") and len(results[0].boxes) else [] #list all detections
    prompt = detections_to_prompt(detections) # create the prompt
    explanation = call_gptoss(prompt) # use the prompt on gpt-oss
    return annotated, explanation

# Run detection + annotate
def detect_webcam(image):
    results = yolo.predict(image)
    return results[0].plot()

with gr.Blocks() as demo:
    gr.Markdown("# ♻️ Trash Detector (YOLO only)")
    
    with gr.Tab("Upload Image"):
        img_in = gr.Image(type="numpy", label="Upload image")
        img_out = gr.Image(type="numpy")
        out_text = gr.Textbox(label='GPT-OSS instructions',lines=6)
        button = gr.Button('Analyze')
        button.click(detect_and_explain, inputs=img_in, outputs=[img_out, out_text])

    with gr.Tab("Webcam"):
        cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        cam_out = gr.Image()
        cam.stream(detect_webcam, inputs=cam, outputs=cam_out)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)