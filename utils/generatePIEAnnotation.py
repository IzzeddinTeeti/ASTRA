import argparse
import pandas as pd
import sys
sys.path.append('.')
from data.PIE_origin import PIE
import os

def generate_PIE_data(root_dir):
    imdb = PIE(data_path=root_dir)
    imdb.generate_database()
    data = pd.read_pickle(os.path.join(root_dir, 'data_cache/pie_database.pkl'))
    df = pd.DataFrame(columns=['set_id', 'video_id', 'ped_id', 'frame_no', 'x_1', 'y_1', 'x_2', 'y_2'])
    for curr_set in data.keys():
        for video_val in data[curr_set].keys():
            for p_id in data[curr_set][video_val]['ped_annotations'].keys():
                set_id, video_id, ped_id = p_id.split('_')
                frames = data[curr_set][video_val]['ped_annotations'][p_id]['frames']
                bbox = data[curr_set][video_val]['ped_annotations'][p_id]['bbox']
                set_idxs = [set_id]* len(frames)
                video_idxs = [video_id]*len(frames)
                ped_ids = [ped_id]*len(frames)
                temp_df = pd.DataFrame(list(zip(set_idxs, video_idxs, ped_ids, frames, bbox)), 
                                    columns=['set_id', 'video_id', 'ped_id', 'frame_no', 'bbox'])
                # df = df.append(temp_df)
                temp_df[['x_1', 'y_1', 'x_2', 'y_2']] = temp_df['bbox'].tolist()
                temp_df.drop(['bbox'], axis=1, inplace=True)
                df = pd.concat([df, temp_df], ignore_index=True)
    df.sort_values(by = ['set_id', 'video_id', 'frame_no', 'ped_id'], axis=0)
    df.to_csv('./datasets/PIE/final_annotations.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = './datasets/PIE')
    args = parser.parse_args()
    generate_PIE_data(args.data_path)