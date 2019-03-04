'''
Rename with padding a list of files in a folder
'''

import os

if __name__ == '__main__':
    
    path = os.path.join(os.environ['HOME'], 'workspace/Maestria/Videos/Output/tmp_output/')
    for filename in os.listdir(path):
        num, ext = filename.split('.')
        num = num.zfill(5)
        new_filename = num + "." + ext
        print("old: {},\t new: {}".format(filename, new_filename))
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))