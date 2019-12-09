import logging
from pathlib import Path

import motmetrics
import numpy as np
import pandas as pd

from vvc import config
from vvc.detector import object_detection
from vvc.format import vvc_format
from vvc.tracker.iou_tracker import IOUTracker
from vvc.tracker.opencv_tracker import OpenCVTracker
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s  %(levelname)s  %(message)s', level=logging.INFO)

if __name__ == '__main__':

    t_name = 'IOUTracker'
    videos = ['MOV_0861.mp4']
    results = pd.DataFrame()

    detector_name = object_detection.Detector.RETINANET
    detector = object_detection.get_detector(detector_name)

    mh = motmetrics.metrics.create()

    trackers = {
        "patient_iou": IOUTracker(iou_threshold=0.5, dectection_threshold=0.8, min_track_len=4, patience=2),
        "boosting": OpenCVTracker(),
    }

    for t_name, tracker in trackers.items():
        vvc = VVC(detector, tracker)

        accs = []
        v_names = []

        for video_name in videos:
            mot_challenge_folder = Path(config.video_folder).joinpath('mot_challenge')
            ground_true_file = mot_challenge_folder.joinpath(Path(video_name).stem + '.txt')
            assert ground_true_file.exists()

            vvc_file = vvc.count(video_name, frame_rate_factor=0.25)

            # To mot challenge
            mot_file = Path(vvc_file).with_suffix('.txt')
            vvc_format.to_mot_challenge(vvc_file=vvc_file, mot_challenge_file=mot_file)

            # Calc mot metrics
            df_gt = motmetrics.io.loadtxt(ground_true_file)
            df_test = motmetrics.io.loadtxt(mot_file)
            acc = motmetrics.utils.compare_to_groundtruth(df_gt, df_test, 'iou', distth=0.5)
            if acc.events['HId'].notnull().sum() > 0:
                accs.append(acc)
                v_names.append(video_name)

        # Summary of mot metrics for run
        summary = mh.compute_many(accs,
                                  metrics=motmetrics.metrics.motchallenge_metrics,
                                  names=v_names,
                                  generate_overall=True)
        logging.info('\n' + motmetrics.io.render_summary(summary,
                                                         namemap=motmetrics.io.motchallenge_metric_names,
                                                         formatters=mh.formatters))
        # Save results
        summary['video'] = summary.index
        summary['detector'] = detector_name
        summary['tracker'] = t_name
        summary = summary.rename(columns=motmetrics.io.motchallenge_metric_names)
        results = results.append(summary, ignore_index=True)

    results.to_csv(Path(config.output_folder).joinpath('all_trackers.csv'), index=False, float_format='%.2f')

