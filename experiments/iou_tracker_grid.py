import logging
from pathlib import Path

import motmetrics
import numpy as np

from vvc import config
from vvc.detector import object_detection
from vvc.format import vvc_format
from vvc.tracker.iou_tracker import IOUTracker
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s  %(message)s', level=logging.INFO)

if __name__ == '__main__':
    videos = ['MOV_0861.mp4']

    for video_name in videos:
        mot_challenge_folder = Path(config.video_folder).joinpath('mot_challenge')
        ground_true_file = mot_challenge_folder.joinpath(Path(video_name).stem + '.txt')
        assert ground_true_file.exists()

        for model in [object_detection.Detector.TINY_YOLO3]:
            detector = object_detection.get_detector(model)
            accs = []
            t_names = []
            for iou_threshold in np.arange(0.3, 0.8, 0.1):
                tracker = IOUTracker(iou_threshold=iou_threshold)
                t_names.append('IOUTracker {}iou'.format(iou_threshold))
                vvc = VVC(detector, tracker)
                vvc_file = vvc.count(video_name, frame_rate_factor=0.5)

                # To mot challenge
                mot_file = Path(vvc_file).with_suffix('.txt')
                vvc_format.to_mot_challenge(vvc_file=vvc_file, mot_challenge_file=mot_file)

                # Calc mot metrics
                df_gt = motmetrics.io.loadtxt(ground_true_file)
                df_test = motmetrics.io.loadtxt(mot_file)
                accs.append(motmetrics.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5))

            metrics = [
                'idf1',
                'idp',
                'idr',
                'recall',
                'precision',
                'num_false_positives',
                'num_misses',
                'num_switches',
                'num_fragmentations',
                'mota',
                'motp'
            ]

            mh = motmetrics.metrics.create()
            summary = mh.compute_many(accs, metrics=metrics, names=t_names, generate_overall=True)
            print(motmetrics.io.render_summary(summary,
                                               namemap=motmetrics.io.motchallenge_metric_names,
                                               formatters=mh.formatters))


            # Print best mot metrics and parameters
