
from vvc.tracker.track import TrackedObject
from vvc.tracker.tracker import Tracker


class IOUTracker(Tracker):
    '''
    classdocs
    '''

    def __init__(self, iou_threshold=0.5, dectection_threshold=0.5, min_track_len=2, patience=5):
        '''
        Constructor
        '''
        super().__init__()
        self.dectection_threshold = dectection_threshold
        self.iou_threshold = iou_threshold
        self.min_track_len = min_track_len
        self.tracks = []
        self.patience = patience
    
    def tracking(self, frame, detections):
        '''
        Process the new frame and return the tracking results
        - detection: list of object detections
        '''
        active_tracks = []

        for detected in detections:
            best_iou = 0
            to_track = None
            box = detected.box
            score = detected.probability
            
            # Compare the detected object with each track
            for track in self.tracks:
                if detected.tag == track.tag:
                    current_iou = self.iou(box, track.boxes[-1])
                    if current_iou > best_iou:
                        best_iou = current_iou
                        to_track = track

            if best_iou >= self.iou_threshold:
                to_track.boxes.append(box)
                to_track.probabilities.append(score)
                to_track.frames_from_last_detection = 0

                self.tracks.remove(to_track)
                active_tracks.append(to_track)

            else:  # If is a one, add to tracks
                name = self.get_next_id(detected.tag)
                track = TrackedObject(name, detected.tag, box, score)
                active_tracks.append(track)
        
        # Keep active track objects
        inactive_tracks = []
        for track in self.tracks:
            if track.frames_from_last_detection < self.patience:
                track.frames_from_last_detection += 1
                if max(track.probabilities) >= self.dectection_threshold and len(track.boxes) >= self.min_track_len:
                    active_tracks.append(track)
                else:
                    inactive_tracks.append(track)
                
        self.tracks = active_tracks + inactive_tracks
        
        return active_tracks
        

