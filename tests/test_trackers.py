
import logging
import unittest

from vvc.detector import object_detection
from vvc.tracker.iou_tracker import IOUTracker
from vvc.tracker.opencv_tracker import OpenCVTracker
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class TrackersTestCase(unittest.TestCase):

    video_name = 'MOV_0861.mp4'
    detector = object_detection.get_detector(object_detection.Detector.TINY_YOLO3)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def __counting(self, tracker):
        vvc = VVC(self.detector, tracker)
        vvc.count(self.video_name, frame_rate_factor=0.2)

    def test_iou_tracker(self):
        tracker = IOUTracker(iou_threshold=0.5, dectection_threshold=0.8, min_track_len=4, patience=2)
        self.__counting(tracker)

    def test_opencv_tracker(self):
        tracker = OpenCVTracker()
        self.__counting(tracker)

