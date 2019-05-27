from tensorflow.python.ops.gen_linalg_ops import self_adjoint_eig


class NaiveTracker():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.tracked_objects = []
        self.last_tags_id = {}
        
    def range_overlap(self, a_min, a_max, b_min, b_max):
        '''
        Check if Neither range is completely greater than the other
        '''
        
        return (a_min <= b_max) and (b_min <= a_max)
        
    def overlap(self, a_box, b_box):
        '''
        Check if the boxes are overlapping
        '''
        (ax0, ay0, ax1, ay1) = a_box
        (bx0, by0, bx1, by1) = b_box
        return self.range_overlap(ax0, ax1, bx0, bx1) and self.range_overlap(ay0, ay1, by0, by1)
    
    def getNextId(self, tag):
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
        '''
        # Deactivate all tracked objects
        for tracked in self.tracked_objects:
            tracked.deactivate()
        
        for detected in detections:
            is_new = True
            
            # Compare detected object with tracked ones
            for tracked in self.tracked_objects:
                if detected.tag == tracked.tag and self.overlap(detected.box, tracked.box):
                    is_new = False
                    tracked.box = detected.box;
                    tracked.active = True
                    tracked.probability = detected.probability
                    break
                
            # If is new add to tracked objects
            if is_new:
                name = self.getNextId(detected.tag)
                tracked = TrackedObject(name, detected.tag, detected.box, detected.probability)
                
                self.tracked_objects.append(tracked)
        
        # Keep active tracked objects
        self.tracked_objects = [x for x in self.tracked_objects if x.active]
        
        return self.tracked_objects

    
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
        self.active = True
        self.probability = probability

    def deactivate(self):
        self.active = False
        
        