import json
import logging
import os

import ffmpeg
import skvideo.io

logger = logging.getLogger(__name__)


class VideoWrapper(object):
    
    def __init__(self, video_path):
        self.video_path = video_path

    def video_reader(self):
        '''
        Return a frame by frame reader
        '''
        inputparameters = {}
        outputparameters = {}
        reader = skvideo.io.FFmpegReader(self.video_path, 
                                         inputdict = inputparameters,
                                         outputdict = outputparameters)
        return reader
    
    def save_to_video(self, output_path, frame_rate):
        """ Save a sequence of images into a mp4 video file """
        
        pattern = os.path.join(output_path, '*.jpg')
        
        # start the FFmpeg writing subprocess with following parameters
        (
            ffmpeg
            .input(pattern, pattern_type='glob', framerate=frame_rate)
            .output(self.video_path, vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run()
        )
    
        
    def avg_frame_rate(self):
        """ Extract the average frame rate from video metadata """
        
        metadata = skvideo.io.ffprobe(self.video_path)
        logger.debug('metadata keys: %s', metadata.keys())
        logger.debug('video metadata: %s', json.dumps(metadata["video"], indent=2))
        
        a, b = metadata['video']['@avg_frame_rate'].split('/')
        avg_frame_rate = float(a) / float(b);
        
        logger.info('avg_frame_rate: %s', avg_frame_rate)
        
        return avg_frame_rate
       
    def total_frames(self):
        """ Extract the number of frames rate from video metadata """
        
        metadata = skvideo.io.ffprobe(self.video_path)
        
        total_frames = metadata['video']['@nb_frames']
        
        logger.info('total_frames: %s', total_frames)
        
        return total_frames
