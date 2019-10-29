'''
'''
import xml.etree.ElementTree as ET

import pandas as pd

def to_mot_challenge(cvat_file, mot_challenge_file):
    ids = dict()
    mot_columns = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
    
    mot = pd.DataFrame(columns=mot_columns + ['conf', 'x', 'y', 'z'])
    
    tree=ET.parse(cvat_file)
    
    
    
    # Consume cvat file
    annotations = tree.getroot() 
    for track in annotations.iter('track'):
        
        if track.attrib['label'] == 'vehicle':
            track_id = track.attrib['id']
            
            for box in track:
                frame_id = int(box.get('frame'))
                
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                                
                row_df = pd.DataFrame([[frame_id, track_id, xtl, ytl, xbr - xtl, ybr - ytl]], 
                                      columns=mot_columns)
                mot = mot.append(row_df, ignore_index=True, sort=False)
                
    mot = mot.sort_values(by=['frame', 'bb_left'])
    
    for index, row in mot.iterrows():
        t_id = row['id']
        if not t_id in ids:
            ids[t_id] = len(ids)
        
        track_id = ids[t_id]
        
        mot.loc[index, 'id'] = track_id
    
    # Process data frame
    
    mot[['conf', 'x', 'y', 'z']] = -1
    
    mot = mot.sort_values(by=['frame'])
    
    mot.to_csv(mot_challenge_file, float_format='%.3f', header=False, index=False)
    
    return mot

