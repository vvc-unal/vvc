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
        'Patient_IOU': IOUTracker(iou_threshold=0.5, dectection_threshold=0.8, min_track_len=4, patience=2),
        'BOOSTING': OpenCVTracker('BOOSTING'),
        'KCF': OpenCVTracker('KCF'),
        'TLD': OpenCVTracker('TLD'),
        'MEDIANFLOW': OpenCVTracker('MEDIANFLOW'),
        'MOSSE': OpenCVTracker('MOSSE'),
        'CSRT': OpenCVTracker('CSRT'),
    }

results_folder = Path(config.output_folder).joinpath('Results')
csv_path = results_folder.joinpath('all_trackers.csv')


def experiment():

    videos = ['MOV_0861.mp4']
    results = pd.DataFrame()

    for detector_name in object_detection.vvc_models:
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

                vvc_file = vvc.count(video_name, frame_rate_factor=0.2)

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

            # Save results
            summary['fps'] = fps
            summary['video'] = summary.index
            summary['detector'] = detector_name.name
            summary['tracker'] = t_name
            summary = summary.rename(columns=motmetrics.io.motchallenge_metric_names)
            results = results.append(summary, ignore_index=True)

            logging.info(results)

    results.to_csv(csv_path, index=False, sep=';', float_format='%.2f', decimal=',')


def plot_results():
    measurements = {'MOTA': {'better': 'higher', 'perfect': '100%', 'title': 'MOTA'},
                    'MOTP': {'better': 'higher', 'perfect': '100%', 'title': 'MOTP'},
                    'MT': {'better': 'higher', 'perfect': '100%', 'title': 'Mostly tracked targets'},
                    'ML': {'better': 'lower', 'perfect': '0%', 'title': 'Mostly lost targets'},
                    'IDs': {'better': 'lower', 'perfect': '0', 'title': 'Identity switches'},
                    'FM': {'better': 'lower', 'perfect': '0', 'title': 'Fragmentations'},
                    'FP': {'better': 'lower', 'perfect': '0', 'title': 'False positives'},
                    'FN': {'better': 'lower', 'perfect': '0', 'title': 'False negatives'}
                    }

    results = pd.read_csv(csv_path, sep=';', decimal=',')

    # preprocessing
    results = results[results['video'] == 'OVERALL']
    results['MOTP'] = (1 - results['MOTP']) * 100
    results['MOTA'] = results['MOTA'] * 100
    results[['MT', 'ML']] = results[['MT', 'ML']].div(results['GT'], axis=0)
    logging.info(results)

    # Plot results
    x_column = 'fps'
    filled_markers = ('o', 'v', '>', 's', 'P', 'X', 'D')

    n = 5
    fig, axes = plt.subplots(4, 2, figsize=(4*n, 3*n))
    ax_index = 0

    for measure, properties in measurements.items():
        ax = axes.reshape(-1)[ax_index]
        ax_index += 1
        marker_index = 0

        for t_name, tracker in trackers.items():
            df = results[results['tracker'] == t_name]
            x = df[x_column]
            y = df[measure]
            ax.scatter(x=x, y=y, label=t_name, marker=filled_markers[marker_index], cmap='Set1')
            for index, row in df.iterrows():
                ax.annotate(row['detector'],
                            xy=(row[x_column], row[measure]),
                            xytext=(0, 6),  # n points vertical offset
                            textcoords="offset points",
                            size=5,
                            ha='center')
            marker_index += 1

        ax.legend(loc='lower left', ncol=2, fontsize='x-small')
        ax.grid(True)

        ax.set_title(properties['title'])
        ylabel = '{} {}'.format(measure, '(%)' if '%' in properties['perfect'] else '')

        if 'lower' == properties['better']:
            ax.invert_yaxis()
            ylabel += ' ({} is better)'.format(properties['better'])

        ax.set_ylabel(ylabel)

        ax.set_xlabel('FPS')

        fig.tight_layout()

        img_path = results_folder.joinpath('{}_vs_fps.png'.format(measure))
        fig.savefig(img_path, dpi=300)


if __name__ == '__main__':
    experiment()
    plot_results()
