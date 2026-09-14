import argparse
import cv2
import sys
import logging
from src.pipeline import FireAIPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def process_image(pipeline, source_path, output_path):
    frame = cv2.imread(source_path)
    if frame is None:
        print(f"Error: Could not read image {source_path}")
        return
    
    detections = pipeline.process_frame(frame)
    annotated = pipeline.annotate_frame(frame, detections)
    
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to {output_path}")

DISPLAY_W = 1280
DISPLAY_H = 720

def process_video(pipeline, source_path, output_path):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {source_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Fit display window to DISPLAY_W x DISPLAY_H keeping aspect ratio
    scale  = min(DISPLAY_W / width, DISPLAY_H / height)
    disp_w = int(width  * scale)
    disp_h = int(height * scale)

    win = "Fire & Smoke Detection  |  Press Q to stop"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)

    print(f"Processing: {source_path}")
    print(f"Resolution: {width}x{height}  |  Display: {disp_w}x{disp_h}  |  FPS: {fps}")
    frame_count = 0

    while True:
        if not cap.grab():
            break
        ret, frame = cap.retrieve()
        if not ret:
            break

        detections = pipeline.process_frame(frame)
        annotated  = pipeline.annotate_frame(frame, detections)
        out.write(annotated)
        frame_count += 1

        display = cv2.resize(annotated, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved annotated video to {output_path}")

def process_webcam(pipeline, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_index}")
        return
        
    print("Starting webcam stream. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        detections = pipeline.process_frame(frame)
        annotated = pipeline.annotate_frame(frame, detections)
        
        cv2.imshow("Fire Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Fire Detection")
    parser.add_argument("--source", type=str, help="Path to input image or video, or 'webcam' for camera")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output (default: output.jpg or output.mp4)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda)")
    
    args = parser.parse_args()
    
    if not args.source:
        parser.print_help()
        sys.exit(1)
        
    pipeline = FireAIPipeline(detector_conf=args.conf, device=args.device, target_fps=30)
    
    if args.source.lower() == "webcam":
        process_webcam(pipeline)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        output = args.output if args.output != "output.jpg" else "output.mp4"
        process_video(pipeline, args.source, output)
    else:
        process_image(pipeline, args.source, args.output)
