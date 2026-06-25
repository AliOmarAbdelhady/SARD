"""
SARD - Search and Rescue Human Detection GUI
Uses trained YOLOv8m model to detect humans in images and videos.

Usage:
    1. Place 'best_sard_yolov8m.pt' in the same directory as this script
    2. pip install ultralytics gradio opencv-python
    3. python SARD_Demo_App.py
    4. Open http://localhost:7860 in your browser
"""

import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_sard_yolov8m.pt")
model = YOLO(MODEL_PATH)
CLASS_NAME = "human"


def detect_image(image, conf_threshold):
    if image is None:
        return None, "No image provided."

    results = model.predict(image, conf=conf_threshold, imgsz=640, verbose=False)
    annotated = results[0].plot()

    boxes = results[0].boxes
    lines = [f"Detected {len(boxes)} {CLASS_NAME}(s)\n"]
    for i, box in enumerate(boxes):
        conf = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        lines.append(
            f"  Person {i + 1}: confidence={conf:.2f}, "
            f"bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
        )

    return annotated, "\n".join(lines)


def detect_video(video_path, conf_threshold, frame_step=3, batch_size=16):
    """Process video with frame skipping and batched inference.

    Args:
        frame_step: Process 1 out of every N frames (3 = every 3rd frame).
        batch_size: Frames sent per model call — higher = better GPU utilization.
    """
    if video_path is None:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_annotated = None
    batch_frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            batch_frames.append(frame)
        else:
            # Reuse last annotated frame for skipped frames
            if last_annotated is not None:
                writer.write(last_annotated)

        # Run batched inference when batch is full
        if len(batch_frames) >= batch_size:
            results = model.predict(batch_frames, conf=conf_threshold, imgsz=640, verbose=False)
            for r in results:
                annotated = r.plot()
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                last_annotated = annotated
                writer.write(annotated)
            batch_frames.clear()

        frame_idx += 1

    # Flush remaining frames in the last partial batch
    if batch_frames:
        results = model.predict(batch_frames, conf=conf_threshold, imgsz=640, verbose=False)
        for r in results:
            annotated = r.plot()
            if annotated.shape[1] != width or annotated.shape[0] != height:
                annotated = cv2.resize(annotated, (width, height))
            writer.write(annotated)

    cap.release()
    writer.release()
    inferred = total_frames // frame_step
    print(f"Done. {total_frames} total frames, {inferred} inferred (every {frame_step}th).")
    return output_path


with gr.Blocks(title="SARD - Search and Rescue Detection") as app:
    gr.Markdown("# Search and Rescue Human Detection")
    gr.Markdown(
        "Upload an image or video to detect humans using the trained YOLOv8m model."
    )

    with gr.Tab("Image Detection"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Upload Image", type="numpy")
                img_conf = gr.Slider(
                    0.1, 0.95, value=0.25, step=0.05, label="Confidence Threshold"
                )
                img_btn = gr.Button("Detect", variant="primary")
            with gr.Column():
                img_output = gr.Image(label="Detection Result", type="numpy")
                img_text = gr.Textbox(label="Detection Details", lines=8)

        img_btn.click(
            fn=detect_image,
            inputs=[img_input, img_conf],
            outputs=[img_output, img_text],
        )

    with gr.Tab("Video Detection"):
        with gr.Row():
            with gr.Column():
                vid_input = gr.Video(label="Upload Video")
                vid_conf = gr.Slider(
                    0.1, 0.95, value=0.25, step=0.05, label="Confidence Threshold"
                )
                vid_step = gr.Slider(
                    1, 10, value=3, step=1,
                    label="Process every Nth frame (higher = faster)",
                )
                vid_btn = gr.Button("Process Video", variant="primary")
            with gr.Column():
                vid_output = gr.Video(label="Annotated Video")

        vid_btn.click(
            fn=detect_video,
            inputs=[vid_input, vid_conf, vid_step],
            outputs=[vid_output],
        )

app.launch(share=False)
