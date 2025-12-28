import cv2
import numpy as np
import pandas as pd
import pickle
import torch
import motmetrics as mm
import os
import time
from byte.byte_tracker import BYTETracker  # Ensure your BYTETracker implementation is available
from IOU_Tracker import IOUTracker         # Ensure your IOUTracker implementation is available

import numpy as np
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a: np.asarray(a, dtype=float)

def get_image_frames(image_dir):
    """
    Reads frames (images) from a directory.
    
    Args:
        image_dir (str): Path to the directory containing images.
    
    Returns:
        list: A list of image frames (numpy arrays).
    """
    image_files = sorted([os.path.join(image_dir, f)
                          for f in os.listdir(image_dir)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print(f"No images found in {image_dir}.")
        return []
    
    frames = []
    for file in image_files:
        frame = cv2.imread(file)
        if frame is None:
            print(f"Error loading image: {file}")
        else:
            frames.append(frame)
    
    return frames

def load_mot_detections(det_path):
    """
    Load MOT format detections from a file.
    
    Args:
        det_path (str): Path to the detection file.
        Expected format per line: [frame,id,x,y,w,h,score]
    
    Returns:
        list: A list of detections, where each detection follows the format:
              [frame, xmin, ymin, xmax, ymax, score, class]
    """
    detections = []
    
    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            try:
                frame = int(parts[0])
                # parts[1] is id (not used for detection)
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = float(parts[6])
            except ValueError:
                continue
            
            # Skip invalid detections (e.g., negative coordinates)
            if x < 0 or y < 0:
                continue
            
            xmax = x + w
            ymax = y + h
            cls = 0  # Default class; adjust if necessary.
            detections.append([frame, x, y, xmax, ymax, score, cls])
    return detections

def real_time_dataset(frames, detections, fps=30):
    """
    Simulate a real-time stream of frames and detections.
    """
    time_per_frame = 1 / fps
    for frame_idx, frame in enumerate(frames):
        # Get detections for the current frame (frames are assumed to be 1-indexed)
        frame_detections = [d for d in detections if d[0] == frame_idx + 1]
        yield frame, frame_detections
        time.sleep(time_per_frame)

def run_tracker(tracker, frames, detections, fps=30):
    """
    Run the tracker (BYTETracker or IOUTracker) on the given frames and detections.
    
    Args:
        tracker: Tracker instance with an update() method.
        frames (list): List of image frames (numpy arrays).
        detections (list): List of detections in format [frame, xmin, ymin, xmax, ymax, score, class].
        fps (int): Frames per second (for simulation timing).
    
    Returns:
        list: Tracked objects in the format:
              [frame, track id, xmin, ymin, width, height, score, class]
    """
    tracked_objects = []
    frame_gen = real_time_dataset(frames, detections, fps)
    
    frame_idx = 0
    while True:
        try:
            frame, frame_detections = next(frame_gen)
        except StopIteration:
            break
        
        # Convert detections: remove frame number, leaving [xmin, ymin, xmax, ymax, score, class]
        dets = [det[1:] for det in frame_detections]
        if len(dets) > 0:
            detection_array = np.array(dets)
        else:
            detection_array = np.empty((0, 6))
        
        # Update tracker with current frame's detections.
        online_tracks = tracker.update(detection_array)
        
        # Each track is expected to be in the format:
        # [xmin, ymin, xmax, ymax, track id, class, score]
        i = 0
        while i < len(online_tracks):
            t = online_tracks[i]
            xmin, ymin, xmax, ymax, track_id, cls, score = t
            w = xmax - xmin
            h = ymax - ymin
            tracked_objects.append([frame_idx + 1, track_id, xmin, ymin, w, h, score, cls])
            i += 1
        
        frame_idx += 1
    
    return tracked_objects

def evaluate_tracking(gt_path, tracked_objects):
    """
    Evaluate tracking performance using MOT metrics.
    
    Args:
        gt_path (str): Path to the ground truth file.
        tracked_objects (list): List of tracked objects in format:
                                [frame, id, x, y, w, h, conf, class]
    
    Returns:
        float: MOTA score.
    """
    gt_data = pd.read_csv(gt_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"])
    print(f"Ground truth entries: {len(gt_data)}")
    gt_data = gt_data[gt_data["conf"] != 0]
    print(f"After filtering conf==0: {len(gt_data)}")
    gt_data = gt_data[gt_data["y"] != -1]
    
    track_df = pd.DataFrame(tracked_objects, columns=["frame", "id", "x", "y", "w", "h", "conf", "class"])
    track_df.to_csv('output.txt', sep=',', index=False)
    
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_data["frame"].unique()):
        gt_frame = gt_data[gt_data["frame"] == frame]
        pred_frame = track_df[track_df["frame"] == frame]
        gt_ids = gt_frame["id"].values
        pred_ids = pred_frame["id"].values
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values
        import numpy as np
        if not hasattr(np, "asfarray"):
            np.asfarray = lambda a: np.asarray(a, dtype=float)
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)
    
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="Overall")
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))
    mota_score = summary.loc["Overall", "mota"]
    return mota_score

if __name__ == "__main__":
    # Update these paths with your actual locations.
    image_path = "train\\MOT17-13-SDP\\img1"  # Folder containing image frames.
    gt_path = "train\\MOT17-13-SDP\\gt\\gt.txt"             # Ground truth file (MOT format).
    det_path = "train\\MOT17-13-SDP\\det\\det.txt"           # Detections file (MOT format)
         # Detections file (MOT format)
    
    frames = get_image_frames(image_path)
    print(f"Loaded {len(frames)} frames from image folder.")
    
    detections = load_mot_detections(det_path)
    print(f"Detections generated: {len(detections)}")
    
    # Instantiate your tracker.
    # For ByteTracker (uncomment the next line if you prefer using it):
    # tracker = BYTETracker(track_thresh=0.45, track_buffer=25, match_thresh=0.8, frame_rate=30)
    # For IOUTracker:
    tracker = BYTETracker(track_thresh=0.45, track_buffer=25, match_thresh=0.8, frame_rate=30)
    
    # Run the tracker over all frames.
    tracked_objects = run_tracker(tracker, frames, detections)
    print(f"Tracking results generated: {len(tracked_objects)}")
    
    # Evaluate tracking performance if ground truth is available.
    if gt_path is not None:
        mota = evaluate_tracking(gt_path, tracked_objects)
        print(f"MOTA Score: {mota * 100:.2f}")
    
    # Optionally, you can pickle the tracker instance if required by your assignment.
    with open("Tracker.pkl", "wb") as f:
        pickle.dump(tracker, f)
    print("Tracker instance saved as 'Tracker.pkl'.")
    tracker = IOUTracker(iou_threshold=0.8)
    
    # Run the tracker over all frames.
    tracked_objects = run_tracker(tracker, frames, detections)
    print(f"Tracking results generated: {len(tracked_objects)}")
    
    # Evaluate tracking performance if ground truth is available.
    if gt_path is not None:
        mota = evaluate_tracking(gt_path, tracked_objects)
        print(f"MOTA Score: {mota * 100:.2f}")
    
    # Optionally, you can pickle the tracker instance if required by your assignment.
    with open("Tracker.pkl", "wb") as f:
        pickle.dump(tracker, f)
    print("Tracker instance saved as 'Tracker.pkl'.")
