import cv2
import skvideo.io
import pandas as pd
import numpy as np
import tqdm


species_list = ['species_fourspot',
 'species_grey sole',
 'species_other',
 'species_plaice',
 'species_summer',
 'species_windowpane',
 'species_winter']
df = pd.read_csv('../../data/training.csv')

def process_row(row,frame,class_name, display=False, bounding_pad=2):
    x1, x2, y1, y2 = map(int, row[['x1', 'x2', 'y1', 'y2']].as_matrix())
    length = row['length']
    if length < 100:
        delta = 50
    elif length >= 100 and length < 150:
        delta = 75
    elif length >= 150 and length < 200:
        delta = 100
    elif length >= 200 and length < 250:
        delta = 125
    elif length >= 250 and length < 300:
        delta = 150
    elif length >= 300 and length < 350:
        delta = 175
    else:
        delta = int(length//2)
        
    delta += bounding_pad
    mid_x = int(x1 + x2)//2
    mid_y = int(y1 + y2)//2
    max_y, max_x, ch = frame.shape
    start_x = max(0, mid_x-delta)
    start_y = max(0, mid_y-delta)
    end_x = min(mid_x+delta, max_x)
    end_y = min(mid_y+delta, max_y)
    
    image = np.copy(frame)
    cropped_image = np.copy(image)[start_y:end_y,start_x:end_x]
    
    if display:
        pass
    else:
        return cropped_image

def process_video(df, video_index,skip=None,display_frames=True):
    if type(video_index)==str:
        x = df[df['video_id']==video_index].dropna()
    else:
        videos = df['video_id'].unique()
        x = df[df['video_id']==videos[video_index]].dropna()
    m = x['video_id'].iloc[0]
    base_path = '../../data/train_videos/'
    x['species'] = x.apply(lambda row: row[species_list].argmax(), axis=1)
    max_frame = x['frame'].max()
    vid_generator = skvideo.io.FFmpegReader(base_path + m + '.mp4')
    counter = -1
    outer_counter = 0
    images = []
    for i, row in x.iterrows():
        target_frame = row['frame']
        for f in vid_generator.nextFrame():
            frame = f
            counter +=1
            if counter == target_frame:
                break
        if skip:
            if outer_counter % skip == 0:
                ret = process_row(row,frame,row['species'], display=display_frames, bounding_pad=20)
        else:
            ret = process_row(row,frame,row['species'], display=display_frames, bounding_pad=20)
        if display_frames == False:
            images.append(ret)
        outer_counter += 1
    return images


total_counter = 0
all_videos = videos = df['video_id'].unique()
base_im_folder = '../../data/binary_classification/fish/'

for video in tqdm.tqdm(all_videos):
    try:
        u = process_video(df, video, None, False)
        for img in u:
            fname = base_im_folder + '{}.png'.format(total_counter)
            total_counter+=1
            cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print("cant process {}".format(video))
        pass



