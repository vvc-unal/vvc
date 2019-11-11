import logging
from pathlib import Path

import motmetrics
import numpy as np
import pandas as pd

from vvc import config
from vvc.detector import object_detection
from vvc.format import vvc_format
from vvc.tracker.iou_tracker import IOUTracker
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s  %(levelname)s  %(message)s', level=logging.INFO)

if __name__ == '__main__':

    t_name = 'IOUTracker'
    videos = ['MOV_0861.mp4']
    results = pd.DataFrame()

    detector_name = object_detection.Detector.RETINANET
    detector = object_detection.get_detector(detector_name)

    mh = motmetrics.metrics.create()

    for iou_threshold in np.arange(0.3, 0.8, 0.1):
        for dectection_threshold in np.arange(0.5, 0.9, 0.1):
            for min_track_len in np.arange(1, 5, 1):
                for patience in np.arange(2, 9, 1):
                    accs = []
                    v_names = []
                    logging.info('Tracker params: iou_t=%.2f, dectection_t=%.2f, min_len=%s, patience=%s',
                                 iou_threshold,
                                 dectection_threshold,
                                 min_track_len,
                                 patience)
                    tracker = IOUTracker(iou_threshold, dectection_threshold, min_track_len, patience)
                    vvc = VVC(detector, tracker)

                    for video_name in videos:
                        mot_challenge_folder = Path(config.video_folder).joinpath('mot_challenge')
                        ground_true_file = mot_challenge_folder.joinpath(Path(video_name).stem + '.txt')
                        assert ground_true_file.exists()

                        vvc_file = vvc.count(video_name, frame_rate_factor=0.5)

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
                    summary['iou_threshold'] = iou_threshold
                    summary['dectection_threshold'] = dectection_threshold
                    summary['min_track_len'] = min_track_len
                    summary['patience'] = patience
                    summary = summary.rename(columns=motmetrics.io.motchallenge_metric_names)
                    results = results.append(summary, ignore_index=True)

            results.to_csv(Path(config.output_folder).joinpath('iou_tracker_grid.csv'),
                           index=False,
                           float_format='%.2f')

