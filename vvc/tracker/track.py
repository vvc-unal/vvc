

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