import logging
from pathlib import Path

import matplotlib.pyplot as plt
import motmetrics
import pandas as pd

from vvc import config
from vvc.detector import object_detection
from vvc.format import vvc_format
from vvc.tracker.iou_tracker import IOUTracker
from vvc.tracker.opencv_tracker import OpenCVTracker
from vvc.vvc import VVC

logging.basicConfig(format='%(asctime)s  %(levelname)s  %(message)s', level=logging.INFO)

trackers = {
        "Patient_IOU": IOUTracker(iou_threshold=0.5, dectection_threshold=0.8, min_track_len=4, patience=2),
        "BOOSTING": OpenCVTracker('BOOSTING'),
        'KCF': OpenCVTracker('KCF'),
        'TLD': OpenCVTracker('TLD'),
        'MEDIANFLOW': OpenCVTracker('MEDIANFLOW'),
        'MOSSE': OpenCVTracker('MOSSE'),
        'CSRT': OpenCVTracker('CSRT'),
    }


def experiment():

    videos = ['MOV_0861.mp4']
    results = pd.DataFrame()

    detector_name = object_detection.Detector.TINY_YOLO3
    detector = object_detection.get_detector(detector_name)

    mh = motmetrics.metrics.create()

    for t_name, tracker in trackers.items():
        logging.info('Detector: {}, Tracker: {}'.format(detector_name.name, t_name))
        vvc = VVC(detector, tracker)

        accs = []
        fps = 0
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

            # Calc fps
            counts, times = vvc_format.to_df(vvc_file)
            total = times['total']
            fps = (total.count() / total.sum()) * 1000

        # Summary of mot metrics for run
        summary = mh.compute_many(accs,
                                  metrics=motmetrics.metrics.motchallenge_metrics,
                                  names=v_names,
                                  generate_overall=True)
        logging.info('\n' + motmetrics.io.render_summary(summary,
                                                         namemap=motmetrics.io.motchallenge_metric_names,
                                                         formatters=mh.formatters))

        # Calc fps
        summary['fps'] = fps

        # Save results
        summary['video'] = summary.index
        summary['detector'] = detector_name.name
        summary['tracker'] = t_name
        summary = summary.rename(columns=motmetrics.io.motchallenge_metric_names)
        results = results.append(summary, ignore_index=True)

    results.to_csv(Path(config.output_folder).joinpath('all_trackers.csv'), index=False, float_format='%.2f')


def plot_results():
    results = pd.read_csv(Path(config.output_folder).joinpath('all_trackers.csv'))
    logging.info(results)
    # Plot results
    fig, ax = plt.subplots()

    x_column = 'fps'

    results['MOTA'] = results['MOTA'] * 100

    for t_name, tracker in trackers.items():
        df = results[results['tracker'] == t_name]
        ax.scatter(x=df[x_column], y=df['MOTA'], label=t_name)

    for i, row in results.iterrows():
        plt.annotate(row["tracker"], xy=(row[x_column], row["MOTA"]))

    #ax.legend()
    ax.grid(True)

    ax.set_ylabel('MOTA (%)')
    ax.set_xlabel('FPS')

    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig(Path(config.output_folder).joinpath('mota_vs_fps.png'), dpi=300)


if __name__ == '__main__':
    experiment()
    plot_results()
