import unittest

import os
from vvc.video_utils import VideoWrapper

base_folder = os.path.join(os.environ['HOME'], 'workspace/Maestria/')
video_folder = os.path.join(base_folder, 'Videos/')
output_folder = os.path.join(video_folder, 'frcnn-restnet50/')
videoName = "MOV_0861.mp4"
input_video_file = os.path.abspath(video_folder + videoName + ".mp4")
output_video_file = os.path.abspath(output_folder + videoName + ".mp4")
img_path = os.path.join(output_folder, 'tmp_input/')
output_path = os.path.join(output_folder, 'tmp_output/')
frame_rate = 30

class TestSaveToVideo(unittest.TestCase):
    """ Test the save to video_utils function """
    
    def test_save_existing_images(self):
        output_video = VideoWrapper(output_video_file)
        output_video.save_to_video(output_path, frame_rate)
        self.assertTrue(True)
        
    def test_get_avg_frame_rate(self):
        input_video = VideoWrapper(input_video_file)
        avg_frame_rate = input_video.get_avg_frame_rate()
        self.assertIsInstance(avg_frame_rate, float)
        self.assertTrue(avg_frame_rate > 0)
                
if __name__ == '__main__':
    unittest.main()
    