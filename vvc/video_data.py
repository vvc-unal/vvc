

class VideoData(object):
    '''
    Class that represents the extracted data from a video
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.video = Video()
        self.frames = {}
        
    def add_frame_data(self):
        frame = FrameData()
        self.frames[len(self.frames)] = frame
        return frame


class Video(object):
    def __init__(self):
        '''
        Constructor
        '''
        self.input_file = ""
        self.output_file = ""
    
    
class FrameData(object):
    def __init__(self):
        '''
        Constructor
        '''
        self.name = ""
        self.objects = []
        self.timestamps = {}
    
    def add_object(self):
        obj = ObjectData()
        self.objects.append(obj)
        return obj

    
class ObjectData(object):
    def __init__(self):
        '''
        Constructor
        '''
        self.name = None
        self.tag = None
        self.box = None
        self.probability = None
    
