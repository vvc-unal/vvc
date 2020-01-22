import copy

import cv2 as cv

from vvc.tracker.track import TrackedObject
from vvc.tracker.tracker import Tracker
from ..utils.bbox import bbox_to_rectangle, rectangle_to_bbox


class OpenCVTracker(Tracker):

    def __init__(self, tracker_type):
        super().__init__(tracker_type)
        self.tracker_type = tracker_type
        self.tracks = []
        self.iou_threshold = 0.5
        self.max_missing_frames = 5

    def __create_tracker(self):
        if self.tracker_type == 'BOOSTING':
            return cv.TrackerBoosting_create()
        if self.tracker_type == 'MIL':
            return cv.TrackerMIL_create()
        if self.tracker_type == 'KCF':
            return cv.TrackerKCF_create()
        if self.tracker_type == 'TLD':
            return cv.TrackerTLD_create()
        if self.tracker_type == 'MEDIANFLOW':
            return cv.TrackerMedianFlow_create()
        if self.tracker_type == 'MOSSE':
            return cv.TrackerMOSSE_create()
        if self.tracker_type == 'CSRT':
            return cv.TrackerCSRT_create()

    def tracking(self, frame, detections):
        active_tracks = []
        inactive_tracks = []

        # Update trackers
        for track in self.tracks:
            ok, rectangle = track.tracker.update(frame)
            if ok:
                bbox = rectangle_to_bbox(rectangle)
                bbox = list(map(int, bbox))
            else:
                # Tracking failure
                bbox = copy.deepcopy(track.boxes[-1])
            track.boxes.append(bbox)
            track.probabilities.append(copy.deepcopy(track.probabilities[-1]))

        # Associate detections with tracks
        for detected in detections:
            best_iou = 0
            to_track = None
            box = detected.box
            score = detected.probability
            tag = detected.tag
            
            # Compare the detected object with each new bbox
            for track in self.tracks:
                if detected.tag == track.tag:
                    last_bbox = track.boxes[-1]
                    current_iou = self.iou(box, last_bbox)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        to_track = track

            if best_iou >= self.iou_threshold:
                to_track.probabilities[-1] = score
                to_track.frames_from_last_detection = 0

                self.tracks.remove(to_track)
                active_tracks.append(to_track)

            else:  # If is a new one, add to tracks
                name = self.get_next_id(tag)
                # Initialize tracker with first frame and bounding box
                tracker = self.__create_tracker()
                rectangle = bbox_to_rectangle(box)
                tracker.init(frame, tuple(rectangle))
                track = OpenCVTrackedObject(tracker, name, tag, box, score)
                active_tracks.append(track)

        # Inactive tracks
        for track in self.tracks:
            track.frames_from_last_detection += 1
            if track.frames_from_last_detection < self.max_missing_frames:
                inactive_tracks.append(track)

        self.tracks = active_tracks + inactive_tracks
        
        return active_tracks
        

class OpenCVTrackedObject(TrackedObject):

    def __init__(self, tracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
