import xml.etree.ElementTree as ElementTree
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vvc import config
from vvc.detector import object_detection
from vvc.format import vvc_format


# In[5]:


dataset_folder = config.video_folder

tracker_name = 'BOOSTING'

video_name = 'MOV_0861'

file_extension = '.json'

cvat_folder = Path(config.video_folder).joinpath('Bog')
vvc_folder = Path(config.output_folder)

test_file = cvat_folder.joinpath(video_name + '.xml')

assert test_file.exists()


# In[7]:

def counts_cvat_to_df(in_file):
    test_annotations = {}

    tree = ElementTree.parse(in_file)
    annotations = tree.getroot()
    
    for track in annotations.iter('track'):
        label = track.attrib['label']
        if label != 'ignore':
            for box in track:
                frame_id = box.get('frame')
                if frame_id not in test_annotations:
                    test_annotations[frame_id] = {}
                tag = label
                
                if tag not in test_annotations[frame_id]:
                    test_annotations[frame_id][tag] = 0
                test_annotations[frame_id][tag] += 1

    df = pd.DataFrame.from_dict(test_annotations, orient='index')
    df = df.fillna(0)
    df = df.set_index(pd.to_numeric(df.index))
    df = df.sort_index(kind='mergesort')
    
    return df


test = counts_cvat_to_df(test_file)
test.iloc[100:].head()

# In[11]:


def calc_precision(test, experiment): 
    
    # Combine using function    
    min_combined = test.combine(experiment, np.minimum, fill_value=0)
        
    exp_combined = test.combine(experiment, lambda s1, s2: s2, fill_value=0)
    
    equal_combined = test.combine(experiment, np.equal, fill_value=False)
    
            
    precision = min_combined.div(exp_combined)
    
    precision[equal_combined == True] = 1

    precision = precision.fillna(0)
    
    
    precision.sort_index(inplace=True, kind='mergesort')

    return precision


def calc_recall(test, experiment): 
    
    # Combine using function    
    min_combined = test.combine(experiment, np.minimum, fill_value=0)
        
    test_combined = test.combine(experiment, lambda s1, s2: s1, fill_value=0)
    
    equal_combined = test.combine(experiment, np.equal, fill_value=False)
    
            
    recall = min_combined.div(test_combined)
    
    recall[equal_combined == True] = 1

    recall = recall.fillna(0)
    
    
    recall.sort_index(inplace=True, kind='mergesort')

    return recall


def plot_precision(experiment, vehicle='car', max_index=500):

    precision = calc_precision(test, experiment)
    recall = calc_recall(test, experiment)

    sub_test = test.loc[test.index <= max_index, vehicle]

    sub_precision = precision.loc[precision.index <= max_index, vehicle]
    sub_recall = recall.loc[recall.index <= max_index, vehicle]

    fig, ax = plt.subplots(figsize=(16, 9))

    plt.title(vehicle)

    ax.plot(sub_test.index, sub_test, label='test', color='r')

    if vehicle in experiment.columns:
        sub_predicted = experiment.loc[experiment.index <= max_index, vehicle]
        ax.plot(sub_predicted.index, sub_predicted, label='experiment', color='b')
    
    ax.legend(loc='upper left')

    ax2 = ax.twinx()

    ax2.plot(sub_precision.index, sub_precision, label='precision', color='g')

    ax2.plot(sub_recall.index, sub_recall, label='recall', color='k')

    ax2.legend()

    plt.tight_layout()

    fig.savefig('./img/{}_precision.png'.format(vehicle), dpi=300)

detector = object_detection.Detector.TINY_YOLO3_TRANSFER
experiment_file = vvc_folder.joinpath(video_name).joinpath(tracker_name).joinpath(detector.value + file_extension)
experiment, times = vvc_format.to_df(experiment_file)
plot_precision(experiment)


def compare_models(test_df, models):

    avg_precision = pd.DataFrame(index=test_df.columns)
    avg_p = pd.DataFrame(columns=models)

    avg_fps = pd.DataFrame(index=['1066x600'])
    avg_time = pd.DataFrame(index=['preprocessing', 'detection', 'tracking', 'postprocessing'])

    for model in models:
        print('model: ', model)
        experiment_file = vvc_folder.joinpath(video_name).joinpath(tracker_name).joinpath(model.value + file_extension)

        assert experiment_file.exists()

        experiment, times = vvc_format.to_df(experiment_file)

        precision = calc_precision(test, experiment)

        mean_precision = precision.mean().to_frame(model)

        avg_precision = avg_precision.join(mean_precision)

        avg_p.loc['avg_p', model] = precision.melt().mean()['value']

        # times

        mean_time = times[avg_time.index].mean().to_frame(model)

        avg_time = avg_time.join(mean_time)

        # fps

        total = times['total']

        fps = (total.count() / total.sum()) * 1000

        avg_fps[model] = fps

        avg_p.loc['balance', model] = avg_p.loc['avg_p', model] * fps
        
    return avg_precision, avg_time, avg_fps, avg_p

# dispay
test = counts_cvat_to_df(test_file)
models = object_detection.all_models

avg_precision, avg_time, avg_fps, avg_p = compare_models(test, models)

print(avg_precision)

print(avg_time)

print(avg_fps)


# In[14]:


print(avg_p)

best_precision_model = avg_p.sort_values(by ='avg_p', axis=1, ascending=False).columns[0]

print('best precision model: ', best_precision_model)

best_balance_model = avg_p.sort_values(by ='balance', axis=1, ascending=False).columns[0]

print('best balance model: ', best_balance_model)


# In[15]:


def plot_avg_precision(avg_precision, prefix='basic', cmap=plt.cm.tab10):
    plt.figure()

    ax = avg_precision.plot.bar(figsize=(16, 9), rot=0, cmap=cmap)

    plt.grid(axis='y')

    ax.set_ylabel('Average Counting Precision')

    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig('./img/{}_avg_precision.png'.format(prefix))
    #fig.savefig('./img/{}avg_precision.tif', dpi=300)

    
plot_avg_precision(avg_precision)


# In[16]:


def plot_avg_time(avg_time, prefix='basic', cmap=plt.cm.tab10):
    plt.figure()

    ax = avg_time.plot.bar(figsize=(16, 9), rot=0, cmap=cmap)

    plt.grid(axis='y')

    ax.set_ylabel('Average frame time (ms)')

    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig('./img/{}_avg_time.png'.format(prefix))


plot_avg_time(avg_time)


# In[17]:


def plot_avg_fps(avg_fps, prefix='basic', cmap=plt.cm.tab10):
    plt.figure()

    ax = avg_fps.plot.barh(figsize=(16, 9), cmap=cmap)

    plt.grid(axis='x')

    ax.set_ylabel('Resolution')
    ax.set_xlabel('FPS')

    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig('./img/{}_avg_fps.png'.format(prefix), dpi=300)


plot_avg_fps(avg_fps)




