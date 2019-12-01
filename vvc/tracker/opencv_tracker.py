import cv2 as cv

from vvc.tracker.track import TrackedObject
from vvc.tracker.tracker import Tracker


class OpenCVTracker(Tracker):

    def __init__(self):
        self.tracks = []
        self.iou_threshold = 0.5
        super().__init__()
    
    def tracking(self, frame, detections):
        active_tracks = []

        next_bboxes = []
        for track in self.tracks:
            tracker = track.tracker
            # Update tracker
            ok, bbox = tracker.update(frame)
            if ok:
                next_bboxes.append(bbox)

        for detected in detections:
            best_iou = 0
            to_track = None
            box = detected.box
            score = detected.probability
            tag = detected.tag
            
            # Compare the detected object with each new bbox
            for track_bbox in next_bboxes:
                current_iou = self.iou(box, track_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    to_track = track

            if best_iou >= self.iou_threshold:
                to_track.boxes.append(track_bbox)
                to_track.probabilities.append(score)
                to_track.frames_from_last_detection = 0

                self.tracks.remove(to_track)
                active_tracks.append(to_track)

            else:  # If is a one, add to tracks
                name = self.get_next_id(tag)
                # Initialize tracker with first frame and bounding box
                tracker = cv.TrackerBoosting_create()
                bbox_with_height = (box[0], box[1], box[2]-box[0], box[3]-box[1])
                tracker.init(frame, bbox_with_height)
                track = OpenCVTrackedObject(tracker, name, tag, box, score)
                active_tracks.append(track)
        
        # Keep active track objects
        inactive_tracks = []
                
        self.tracks = active_tracks + inactive_tracks
        
        return active_tracks
        

class OpenCVTrackedObject(TrackedObject):

    def __init__(self, tracker, *args, **kwargs):
        self.tracker = tracker
        super().__init__(*args, **kwargs)

