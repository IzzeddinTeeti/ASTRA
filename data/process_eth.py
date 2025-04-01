import os, random, numpy as np, copy
import sys
sys.path.append('.')
from utils.visualizer import world_to_pixel
from eth import data_generator

import torch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--subset', type=str, default='eth',
                    help='Name of the subset.')
    parser.add_argument('--total_num', type=int, default=5,
                help='Name of the subset.')
    parser.add_argument('--data_path', type=str, default = './datasets/eth_ucy')
    args = parser.parse_args()
    past_length = 8
    future_length = 12
    scale = 1

    if args.subset == 'univ':
        args.total_num = 2
    elif args.subset == 'zara01':
        args.total_num = 3
    else:
        args.total_num = 1

    generator_train = data_generator(args.subset, past_length, future_length,
                                     args.data_path, scale, split='train', phase='training')
    generator_test = data_generator(args.subset, past_length, future_length, 
                                    args.data_path, scale, split='test', 
                                    phase='testing')
    data_to_idx = {'eth':0, 'hotel':1, 'students003':2, 'zara01':3, 'zara02':4, 
                   # Following Data Video not present
                   'zara03':3, 'students001':2, 'uni':2} 
    # Students have same homography matrix as univ data and zara03 homography matrix is not yet present
    homography_matrices = []
    base_path = os.path.join('.', 'datasets', 'eth_ucy', 'homography')
    for data_name, idx in data_to_idx.items():
        if idx < len(homography_matrices):
            continue 
        file_name = os.path.join(base_path, f'H_{data_name}.txt')
        matrix = np.loadtxt(file_name)
        homography_matrices.append(matrix)
    homography_matrices = np.stack(homography_matrices)
    total_num = args.total_num
    all_past_data = [] #(B,N,T,2)
    all_future_data = [] #(B,N,T,2)
    all_valid_num = []
    thres = 5
    print('start process training data:')
    while not generator_train.is_epoch_end():
        data = generator_train()
        if data is not None:
            # Setting Data index and homography matrix for this data
            data_name = data['seq'].split('_')[-2]
            # Since naming convention of uni data is different
            if data_name == 'examples':
                data_name = 'uni'
            data_idx = data_to_idx[data_name]
            homography_matrix = homography_matrices[data_idx]
            last_frame = data['frame']

            loc = torch.stack(data['pre_motion_3D'],dim=0)
            loc_end = torch.stack(data['fut_motion_3D'],dim=0)
            length = loc.shape[1]
            length_f = loc_end.shape[1]
            agent_num = loc.shape[0]
            loc = np.array(loc)
            loc_end = np.array(loc_end)

            # For appending to form data (data_idx, frame_no, pixel_row, pixel_col, x, y)
            pixels_coords = world_to_pixel(loc, homography_matrix, data_name, scale)
            pixels_coords_end = world_to_pixel(loc_end, homography_matrix, data_name, scale)
            frame_nos_past = np.arange(start = last_frame-past_length+1 ,stop=last_frame+1)
            frame_nos_future = np.arange(start = last_frame + 1, stop = last_frame + future_length + 1)
            frame_nos_past = frame_nos_past.reshape(-1,1)[None].repeat(agent_num, axis = 0)
            frame_nos_future = frame_nos_future.reshape(-1,1)[None].repeat(agent_num, axis = 0)
            data_idx_arr_past = data_idx*np.ones(past_length)
            data_idx_arr_fut = data_idx*np.ones(future_length)
            data_idx_arr_past = data_idx_arr_past.reshape(-1,1)[None].repeat(agent_num, axis = 0)
            data_idx_arr_fut = data_idx_arr_fut.reshape(-1,1)[None].repeat(agent_num, axis = 0)

            loc = np.concatenate((data_idx_arr_past, frame_nos_past, pixels_coords, loc), axis = -1)
            loc_end = np.concatenate((data_idx_arr_fut, frame_nos_future, pixels_coords_end, loc_end), axis = -1)
            
            # Flip x and y
            # if data_name not in ('eth', 'hotel'):
            #     temp = loc[...,-1].copy()
            #     loc[...,-1] = loc[...,-2]
            #     loc[...,-2] = temp
            #     temp = loc_end[...,-1].copy()
            #     loc_end[...,-1] = loc_end[...,-2]
            #     loc_end[...,-2] = temp  
                
            # Since the video of following data does not exist
            if data_name in ('zara03', 'uni', 'students001'):
                loc[..., 1] = -1
                loc_end[..., 1] = -1
            feature_size = loc.shape[-1]
            if loc.shape[0] < total_num:
                for i in range(loc.shape[0]):
                    temp = np.zeros((total_num,length,feature_size))
                    temp[0] = loc[i]
                    temp[1:agent_num] = np.delete(loc,i,axis=0)
                    all_past_data.append(temp[None])
                    
                    temp = np.zeros((total_num,length_f,feature_size))
                    temp[0] = loc_end[i]
                    temp[1:agent_num] = np.delete(loc_end,i,axis=0)
                    all_future_data.append(temp[None])
                    all_valid_num.append(agent_num)
            else:
                for i in range(loc.shape[0]):
                    distance_i = np.linalg.norm(loc[:,-1, -2:] - loc[i:i+1,-1, -2:],axis=-1)
                    num_neighbor = np.sum((distance_i < thres).astype(int))
                    if num_neighbor < total_num:
                        temp = np.zeros((total_num,length,feature_size))
                        neighbors_idx = np.argsort(distance_i)
                        neighbors_idx = neighbors_idx[:num_neighbor]
                        temp[:num_neighbor] = loc[neighbors_idx]
                        all_past_data.append(temp[None])

                        temp = np.zeros((total_num,length_f,feature_size))
                        neighbors_idx = neighbors_idx[:num_neighbor]
                        temp[:num_neighbor] = loc_end[neighbors_idx]
                        all_future_data.append(temp[None])
                        all_valid_num.append(num_neighbor)
                    else:
                        neighbors_idx = np.argsort(distance_i)
                        # print('neighbors_idx', neighbors_idx.shape, neighbors_idx)
                        assert neighbors_idx[0] == i
                        neighbors_idx = neighbors_idx[:total_num]
                        temp = loc[neighbors_idx]
                        # print('all_past_data before', len(all_past_data))
                        all_past_data.append(temp[None])
                        # print('all_past_data after', len(all_past_data))
                        temp = loc_end[neighbors_idx]
                        all_future_data.append(temp[None])
                        all_valid_num.append(total_num)

    all_past_data = np.concatenate(all_past_data,axis=0)
    all_future_data = np.concatenate(all_future_data,axis=0)
    print('all_past_data', all_past_data.shape)
    print('all_future_data', all_future_data.shape)
    all_data = np.concatenate([all_past_data,all_future_data],axis=2)
    all_valid_num = np.array(all_valid_num)
    print('all_data', all_data.shape)
    print('all_valid_num', all_valid_num.shape)

    output_folder = os.path.join(args.data_path ,'processed_data_diverse')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")
    else:
        print(f"Folder '{output_folder}' already exists.")

    np.save(output_folder+'/'+ args.subset +'_data_train.npy',all_data)
    np.save(output_folder+'/'+ args.subset +'_num_train.npy',all_valid_num)

                    
    all_past_data = [] #(B,N,T,2)
    all_future_data = [] #(B,N,T,2)
    all_valid_num = []
    print('start process testing data:')
    while not generator_test.is_epoch_end():
        data = generator_test()
        if data is not None:
            # Setting Data index and homography matrix for this data
            data_name = data['seq'].split('_')[-1]
            # Since naming convention of uni data is different
            if data_name == 'examples':
                data_name = 'uni'
            data_idx = data_to_idx[data_name]
            homography_matrix = homography_matrices[data_idx]
            last_frame = data['frame']

            loc = torch.stack(data['pre_motion_3D'],dim=0)
            loc_end = torch.stack(data['fut_motion_3D'],dim=0)
            length = loc.shape[1]
            length_f = loc_end.shape[1]
            agent_num = loc.shape[0]
            loc = np.array(loc)
            loc_end = np.array(loc_end)

            
            # For appending to form data (data_idx, frame_no, pixel_row, pixel_col, x, y)
            pixels_coords = world_to_pixel(loc, homography_matrix, data_name, scale)
            pixels_coords_end = world_to_pixel(loc_end, homography_matrix, data_name, scale)
            frame_nos_past = np.arange(start = last_frame-past_length+1 ,stop=last_frame+1)
            frame_nos_future = np.arange(start = last_frame + 1, stop = last_frame + future_length + 1)
            frame_nos_past = frame_nos_past.reshape(-1,1)[None].repeat(agent_num, axis = 0)
            frame_nos_future = frame_nos_future.reshape(-1,1)[None].repeat(agent_num, axis = 0)
            data_idx_arr_past = data_idx*np.ones(past_length)
            data_idx_arr_fut = data_idx*np.ones(future_length)
            data_idx_arr_past = data_idx_arr_past.reshape(-1,1)[None].repeat(agent_num, axis = 0)
            data_idx_arr_fut = data_idx_arr_fut.reshape(-1,1)[None].repeat(agent_num, axis = 0)

            loc = np.concatenate((data_idx_arr_past, frame_nos_past, pixels_coords, loc), axis = -1)
            loc_end = np.concatenate((data_idx_arr_fut, frame_nos_future, pixels_coords_end, loc_end), axis = -1)
            
            # Flip x and y
            # if data_name not in ('eth', 'hotel'):
            #     temp = loc[...,-1].copy()
            #     loc[...,-1] = loc[...,-2]
            #     loc[...,-2] = temp
            #     temp = loc_end[...,-1].copy()
            #     loc_end[...,-1] = loc_end[...,-2]
            #     loc_end[...,-2] = temp 
                
            # Since the video of following data does not exist
            if data_name in ('zara03', 'uni', 'students001'):
                loc[..., 1] = -1
                loc_end[..., 1] = -1
            feature_size = loc.shape[-1]
            if loc.shape[0] < total_num:
                for i in range(loc.shape[0]):
                    temp = np.zeros((total_num,length,feature_size))
                    temp[0] = loc[i]
                    temp[1:agent_num] = np.delete(loc,i,axis=0)
                    all_past_data.append(temp[None])
                    
                    temp = np.zeros((total_num,length_f,feature_size))
                    temp[0] = loc_end[i]
                    temp[1:agent_num] = np.delete(loc_end,i,axis=0)
                    all_future_data.append(temp[None])
                    all_valid_num.append(agent_num)
            else:
                for i in range(loc.shape[0]):
                    distance_i = np.linalg.norm(loc[:,-1, -2:] - loc[i:i+1,-1, -2:],axis=-1)
                    num_neighbor = np.sum((distance_i < thres).astype(int))
                    if num_neighbor < total_num:
                        temp = np.zeros((total_num,length,feature_size))
                        neighbors_idx = np.argsort(distance_i)
                        neighbors_idx = neighbors_idx[:num_neighbor]
                        temp[:num_neighbor] = loc[neighbors_idx]
                        all_past_data.append(temp[None])

                        temp = np.zeros((total_num,length_f,feature_size))
                        neighbors_idx = neighbors_idx[:num_neighbor]
                        temp[:num_neighbor] = loc_end[neighbors_idx]
                        all_future_data.append(temp[None])
                        all_valid_num.append(num_neighbor)
                    else:
                        neighbors_idx = np.argsort(distance_i)
                        assert neighbors_idx[0] == i
                        neighbors_idx = neighbors_idx[:total_num]
                        temp = loc[neighbors_idx]
                        all_past_data.append(temp[None])
                        temp = loc_end[neighbors_idx]
                        all_future_data.append(temp[None])
                        all_valid_num.append(total_num)

    all_past_data = np.concatenate(all_past_data,axis=0)
    all_future_data = np.concatenate(all_future_data,axis=0)
    print('all_past_data', all_past_data.shape)
    print('all_future_data', all_future_data.shape)
    all_data = np.concatenate([all_past_data,all_future_data],axis=2)
    all_valid_num = np.array(all_valid_num)
    print('all_data', all_data.shape)
    print('all_valid_num', all_valid_num.shape)
    np.save(output_folder+'/'+ args.subset +'_data_test.npy',all_data)
    np.save(output_folder+'/'+ args.subset +'_num_test.npy',all_valid_num)
