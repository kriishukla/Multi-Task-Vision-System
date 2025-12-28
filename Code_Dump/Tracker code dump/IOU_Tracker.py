import numpy as np

class IOUTracker:
    def __init__(self, iou_threshold=0.7):
        # Initialize the IOU threshold, list to hold active trackers, and next track ID.
        self.iou_threshold = iou_threshold
        self.trackers = []  # Each tracker: [xmin, ymin, xmax, ymax, track id, class, score]
        self.next_id = 0

    def _compute_iou(self, box1, box2):
        """
        Compute the IOU between two bounding boxes.
        Each box is expected to be in [xmin, ymin, xmax, ymax] format.
        Returns the IOU score (intersection / union).
        """
        # Determine the coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        # Compute the area of intersection rectangle
        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        interArea = inter_width * inter_height
        
        # Compute the area of both the prediction and ground-truth rectangles
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the union area
        unionArea = box1_area + box2_area - interArea
        
        # Avoid division by zero
        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def update(self, detections):
        """
        Update the trackers with new detections.
        
        - Perform greedy matching using IOU: for each existing tracker, compare with all new detections.
        - If IOU > threshold, match the detection to the tracker.
        - If no match is found for a detection, create a new tracker.
        - Remove trackers that don't match any detection from the previous frame.
        
        Input Format for each detection:
            [xmin, ymin, xmax, ymax, score, class]
        Returns the updated list of trackers in the format:
            [xmin, ymin, xmax, ymax, track id, class, score]
        """
        updated_trackers = []
        matched_trackers = set()
        matched_detections = set()
        candidate_matches = []
        
        # If no existing trackers, initialize all detections as new trackers.
        if len(self.trackers) == 0:
            i = 0
            while i < len(detections):
                det = detections[i]
                new_tracker = [det[0], det[1], det[2], det[3], self.next_id, det[5], det[4]]
                self.next_id += 1
                updated_trackers.append(new_tracker)
                i += 1
            self.trackers = updated_trackers
            return self.trackers
        
        # Build candidate matches: for each tracker and detection compute IOU.
        t_idx = 0
        while t_idx < len(self.trackers):
            tracker = self.trackers[t_idx]
            tracker_box = tracker[0:4]
            d_idx = 0
            while d_idx < len(detections):
                det = detections[d_idx]
                det_box = det[0:4]
                iou = self._compute_iou(tracker_box, det_box)
                if iou > self.iou_threshold:
                    candidate_matches.append((t_idx, d_idx, iou))
                d_idx += 1
            t_idx += 1
                    
        # Sort candidate matches by descending IOU (greedy matching)
        candidate_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Greedily assign detections to trackers (each tracker and detection can match only once)
        i = 0
        while i < len(candidate_matches):
            t_idx, d_idx, iou = candidate_matches[i]
            if t_idx not in matched_trackers and d_idx not in matched_detections:
                det = detections[d_idx]
                # Update tracker with new detection data.
                self.trackers[t_idx] = [det[0], det[1], det[2], det[3], self.trackers[t_idx][4], det[5], det[4]]
                matched_trackers.add(t_idx)
                matched_detections.add(d_idx)
            i += 1
        
        # Keep only trackers that got a match
        i = 0
        while i < len(matched_trackers):
            t_idx = list(matched_trackers)[i]
            updated_trackers.append(self.trackers[t_idx])
            i += 1
        
        # For unmatched detections, create new trackers.
        d_idx = 0
        while d_idx < len(detections):
            if d_idx not in matched_detections:
                det = detections[d_idx]
                new_tracker = [det[0], det[1], det[2], det[3], self.next_id, det[5], det[4]]
                self.next_id += 1
                updated_trackers.append(new_tracker)
            d_idx += 1
        
        # Update the tracker list with the new set of trackers.
        self.trackers = updated_trackers
        return self.trackers