'''
'''
import pandas as pd

from vvc import json_utils
from pandas.tseries.holiday import AbstractHolidayCalendar

def to_mot_challenge(vvc_file, mot_challenge_file):
    video_data = json_utils.load_from_json(vvc_file)
    
    main_columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
    
    df = pd.DataFrame(columns=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    
    # Process video data
    
    ids = dict()
    
    for frame_id, frame_data in video_data.frames.items():
        for track_data in frame_data.tracks:
            
            if not track_data.id in ids:
                ids[track_data.id] = len(ids)
            
            track_id = ids[track_data.id]
            
            box = track_data.box
            bb_left = box[0]
            bb_top = box[1]
            bb_width = box[2] - box[0]
            bb_height = box[3] - box[1]
            
            row_df = pd.DataFrame([[frame_id, track_id, bb_left, bb_top, bb_width, bb_height]], 
                                          columns=main_columns)
            df = df.append(row_df, ignore_index = True, sort=False)
    
    # Process data frame
    
    df[['conf', 'x', 'y', 'z']] = -1
    
    df[main_columns] = df[main_columns].apply(pd.to_numeric, errors='coerce')
    
    df = df.sort_values(by=['frame'])
    
    df.to_csv(mot_challenge_file, float_format='%.3f', header=False, index=False)
    
    return df