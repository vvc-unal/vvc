import copy

from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig


class IOUTracker:
    '''
    classdocs
    '''

    def __init__(self, iou_threshold=0.5, dectection_threshold=0.5, min_track_len=2, patience=5):
        '''
        Constructor
        '''
        self.dectection_threshold = dectection_threshold
        self.iou_threshold = iou_threshold
        self.min_track_len = min_track_len
        self.tracks = []
        self.last_tags_id = {}
        self.patience = patience
        
    @staticmethod
    def iou(a_box, b_box):
        '''
        Check if the boxes are overlapping
        '''
        (ax0, ay0, ax1, ay1) = a_box
        (bx0, by0, bx1, by1) = b_box
        
        x0 = max(ax0, bx0)
        y0 = max(ay0, by0)
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        
        inter_area = max(0, x1 - x0) * max(0, y1 - y0)
        
        a_area = (ax1 - ax0) * (ay1 - ay0)
        b_area = (ax1 - ax0) * (ay1 - ay0)
        
        iou = inter_area / (a_area + b_area - inter_area) 
                
        return iou
    
    def get_next_id(self, tag):
        '''
        Get the next id for the Tag
        '''
        if tag in self.last_tags_id:
            self.last_tags_id[tag] += 1
        else:
            self.last_tags_id[tag] = 1
        return tag + " " + str(self.last_tags_id[tag])
    
    def tracking(self, detections):
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
            if track.frames_from_last_detection <= self.patience:
                track.frames_from_last_detection += 1
                if max(track.probabilities) >= self.dectection_threshold and len(track.boxes) >= self.min_track_len:
                    active_tracks.append(track)
                else:
                    inactive_tracks.append(track)
                
        self.tracks = active_tracks + inactive_tracks
        
        return active_tracks

    
class TrackedObject:
    '''
    Class representing tracked objects
    '''

    def __init__(self, name, tag, box, probability):
        '''
        Constructor
        '''
        self.name = name
        self.tag = tag
        self.boxes = [box]
        self.frames_from_last_detection = 0
        self.probabilities = [probability]
        

