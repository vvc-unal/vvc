import cv2 as cv

from vvc.tracker.track import TrackedObject
from vvc.tracker.tracker import Tracker
from ..utils.bbox import bbox_to_rectangle, rectangle_to_bbox


class OpenCVTracker(Tracker):

    def __init__(self):
        self.tracks = []
        self.iou_threshold = 0.5
        super().__init__()
    
    def tracking(self, frame, detections):
        active_tracks = []
        # Keep active track objects
        inactive_tracks = []

        for detected in detections:
            best_iou = 0
            to_track = None
            box = detected.box
            bbox = None
            score = detected.probability
            tag = detected.tag
            
            # Compare the detected object with each new bbox
            for track in self.tracks:
                if detected.tag == track.tag:
                    # Update tracker
                    ok, rectangle = track.tracker.update(frame)
                    if ok:
                        bbox = rectangle_to_bbox(rectangle)
                        bbox = list(map(int, bbox))
                        current_iou = self.iou(box, bbox)
                        if current_iou > best_iou:
                            best_iou = current_iou
                            to_track = track
                    else:
                        # Tracking failure
                        track.frames_from_last_detection += 1
                        inactive_tracks.append(track)

            if best_iou >= self.iou_threshold:
                to_track.boxes.append(bbox)
                to_track.probabilities.append(score)
                to_track.frames_from_last_detection = 0

                self.tracks.remove(to_track)
                active_tracks.append(to_track)

            else:  # If is a one, add to tracks
                name = self.get_next_id(tag)
                # Initialize tracker with first frame and bounding box
                tracker = cv.TrackerBoosting_create()
                rectangle = bbox_to_rectangle(box)
                tracker.init(frame, tuple(rectangle))
                track = OpenCVTrackedObject(tracker, name, tag, box, score)
                active_tracks.append(track)
                
        self.tracks = active_tracks + inactive_tracks
        
        return active_tracks
        

class OpenCVTrackedObject(TrackedObject):

    def __init__(self, tracker, *args, **kwargs):
        self.tracker = tracker
        super().__init__(*args, **kwargs)

