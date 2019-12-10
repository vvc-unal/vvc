
import json
import pandas as pd

from vvc import json_utils


def to_df(json_file):
    count_summary = {}
    time_summary = {}

    with open(json_file) as json_data:
        data = json.load(json_data)
        for frame_id, objects in data['frames'].items():

            # Extract counts
            if frame_id not in count_summary:
                count_summary[frame_id] = {}

            for obj in objects['objects']:
                tag = obj['tag']

                if tag not in count_summary[frame_id]:
                    count_summary[frame_id][tag] = 0

                count_summary[frame_id][tag] += 1

            # Extract running time
            if frame_id not in time_summary:
                time_summary[frame_id] = {}

            for key, value in objects['timestamps'].items():
                time_summary[frame_id][key] = value

    df = pd.DataFrame.from_dict(count_summary, orient='index')
    df = df.fillna(0)
    df = df.set_index(pd.to_numeric(df.index))
    df = df.sort_index(kind='mergesort')
    df = df.reindex(sorted(df.columns), axis=1)

    exp = pd.DataFrame()

    # Set the values for each perspective
    for column in df.columns:
        for fb_side in ['front', 'back']:
            for lr_side in ['left', 'right']:
                tag = column  # + '_' + fb_side + '_' + lr_side
                exp[tag] = df[column]

    exp = exp.sort_index(kind='mergesort')

    times = pd.DataFrame.from_dict(time_summary, orient='index')
    times['total'] = times.sum(axis=1)

    return exp, times


def to_mot_challenge(vvc_file, mot_challenge_file):
    video_data = json_utils.load_from_json(vvc_file)
    
    main_columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
    
    df = pd.DataFrame(columns=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    
    # Process video data
    
    ids = dict()
    
    for frame_id, frame_data in video_data.frames.items():
        for track_data in frame_data.tracks:
            
            track_id = track_data.id
            
            box = track_data.box
            bb_left = box[0]
            bb_top = box[1]
            bb_width = box[2] - box[0]
            bb_height = box[3] - box[1]
            
            row_df = pd.DataFrame([[frame_id, track_id, bb_left, bb_top, bb_width, bb_height]],
                                  columns=main_columns)
            df = df.append(row_df, ignore_index=True, sort=False)
            
    df = df.sort_values(by=['frame', 'bb_left'])
    
    for index, row in df.iterrows():
        t_id = row['id']
        if t_id not in ids:
            ids[t_id] = len(ids)
        
        track_id = ids[t_id]
        
        df.loc[index, 'id'] = track_id
    
    # Process data frame
    
    df[['conf', 'x', 'y', 'z']] = -1
    
    df[main_columns] = df[main_columns].apply(pd.to_numeric, errors='coerce')
    
    df = df.sort_values(by=['frame'])
    
    df.to_csv(mot_challenge_file, float_format='%.3f', header=False, index=False)
    
    return df