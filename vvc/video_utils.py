import glob
import os

import ffmpeg
import skvideo.io
import json


def video_to_images(input_video_file, img_path):
    """ Convert the video file into a sequence of images """
    counter = 0
    
    videodata = skvideo.io.vreader(input_video_file)
    
    for frame in videodata:
        skvideo.io.vwrite(os.path.join(img_path, str(counter) + '.jpg'), frame)
        counter = counter + 1
    

def video_reader(input_video_file):
    '''
    Return a frame by frame reader
    '''
    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader(input_video_file, 
                                     inputdict = inputparameters,
                                     outputdict = outputparameters)
    return reader

def save_to_video(output_path, output_video_file, frame_rate):
    """ Save a sequence of images into a mp4 video file """
    
    pattern = output_path + '*.jpg'
    
    # start the FFmpeg writing subprocess with following parameters
    (
        ffmpeg
        .input(pattern, pattern_type='glob', framerate=frame_rate)
        .output(output_video_file, vcodec='libx264', pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )


def save_to_video_skvideo(output_path, output_video_file, frame_rate):
    """ Save a sequence of images into a mp4 video file """
    
    # Find and sort images
    list_files = glob.glob(output_path + '*.jpg')
    list_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0] ) )
    
    # start the FFmpeg writing subprocess with following parameters
    inputparameters = {}
    outputparameters = {'-vcodec': 'libx264', "-r": str(frame_rate)}
    writer = skvideo.io.FFmpegWriter(output_video_file,
                                     inputdict = inputparameters,
                                     outputdict = outputparameters,
                                     verbosity=1)

    for file in list_files:
        frame = skvideo.io.vread(file)
        writer.writeFrame(frame)
        
    writer.close()
    
def get_avg_frame_rate(video_path):
    """ Extract the average frame rate from video metadata """
    
    metadata = skvideo.io.ffprobe(video_path)
    print(metadata.keys())
    print(json.dumps(metadata["video"], indent=2))
    
    a, b = metadata['video']['@avg_frame_rate'].split('/')
    avg_frame_rate = float(a) / float(b);
    
    print("avg_frame_rate: ", avg_frame_rate)
    return avg_frame_rate
