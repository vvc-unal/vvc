'''
'''
import xml.etree.ElementTree as ET

import pandas as pd

def to_mot_challenge(cvat_file, mot_challenge_file):
    df = pd.DataFrame(columns=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
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
                                      columns=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height'])
                df = df.append(row_df, ignore_index = True, sort=False)
    
    # Process data frame
    
    df[['conf', 'x', 'y', 'z']] = -1
    
    df = df.sort_values(by=['frame'])
    
    df.to_csv(mot_challenge_file, float_format='%.3f', header=False, index=False)
    
    return df