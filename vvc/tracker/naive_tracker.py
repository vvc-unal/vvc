import copy

from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig


class NaiveTracker():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.tracks = []
        self.last_tags_id = {}
        
    def iou(self, a_box, b_box):
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
        n = 5
        
        for detected in reversed(detections):
            best_iou = 0
            to_track = None
            
            # Compare the detected object with each track
            for track in self.tracks:
                if detected.tag == track.tag:
                    current_iou = self.iou(detected.box, track.box) 
                    if current_iou > best_iou:
                        best_iou = current_iou
                        to_track = track
            
            if best_iou > 0:
                to_track.box = detected.box
                to_track.probability = detected.probability
                to_track.frames_from_last_detection = 0
                
                detections.remove(detected)
                self.tracks.remove(to_track)
                active_tracks.append(to_track)
        
        # If is new add to tracks
        for detected in detections:
            name = self.get_next_id(detected.tag)
            track = TrackedObject(name, detected.tag, detected.box, detected.probability)
            active_tracks.append(track)
        
        # Keep active track objects
        for track in self.tracks:
            if track.frames_from_last_detection < n:
                track.frames_from_last_detection += 1
                active_tracks.append(track)
                
        self.tracks = active_tracks
        
        return self.tracks

    
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
        self.box = box
        self.frames_from_last_detection = 0
        self.probability = probability
        
        
