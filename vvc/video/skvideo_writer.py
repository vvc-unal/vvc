
import skvideo.io

'''
Created on 16/06/2019

@author: juan
'''

class SKVideoWriter(object):
    '''
    classdocs
    '''


    def __init__(self, output_file, frame_rate):
        '''
        Constructor
        '''
        # start the FFmpeg writing subprocess with following parameters
        input_parameters = {'-r': str(frame_rate)}
        output_parameters = {'-vcodec': 'libx264',
                            '-pix_fmt': 'yuv420p', 
                            '-r': str(frame_rate)}
        
        self.writer = skvideo.io.FFmpegWriter(output_file,
                                     inputdict = input_parameters,
                                     outputdict = output_parameters,
                                     verbosity=0)
        
    def writeFrame(self, frame):
        """ Save a sequence of images into a mp4 video file """
        self.writer.writeFrame(frame)
        
        
    def close(self):
        self.writer.close()