import os, random, numpy as np, copy

import sys 
# sys.path.append("..")
from data.eth_preprocessor import preprocess
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils.visualizer import world_to_pixel

def get_ethucy_split(dataset):
     seqs = [
          'biwi_eth',
          'biwi_hotel',
          'crowds_zara01',
          'crowds_zara02',
          'crowds_zara03',
          'students001',
          'students003',
          'uni_examples'
     ]
     if dataset == 'eth':
          test = ['biwi_eth']
     elif dataset == 'hotel':
          test = ['biwi_hotel']
     elif dataset == 'zara01':
          test = ['crowds_zara01']
     elif dataset == 'zara02':
          test = ['crowds_zara02']
     elif dataset == 'univ':
          test = ['students001','students003']

     train, val = [], []
     for seq in seqs:
          if seq in test:
               continue
          train.append(f'{seq}_train')
          val.append(f'{seq}_val')
     return train, val, test

def print_log(print_str, log, same_line=False, display=True):
	'''
	print a string to a log file
	parameters:
		print_str:          a string to print
		log:                a opened file to save the log
		same_line:          True if we want to print the string without a new next line
		display:            False if we want to disable to print the string onto the terminal
	'''
	if display:
		if same_line: print('{}'.format(print_str), end='')
		else: print('{}'.format(print_str))

	# if same_line: log.write('{}'.format(print_str))
	# else: log.write('{}\n'.format(print_str))
	# log.flush()


class ETH_dataset(object):
    def __init__(self, cfg, mode='training', img_transforms = None):
        # file_dir = 'eth_ucy/processed_data_diverse/'
        file_dir = 'datasets/eth_ucy/processed_data_diverse'
        
        dataset = cfg.SUBSET
        if mode == 'training':
            data_file_path = os.path.join(file_dir, dataset +'_data_train.npy')
            num_file_path = os.path.join(file_dir, dataset +'_num_train.npy')
        elif mode == 'testing':
            data_file_path = os.path.join(file_dir, dataset +'_data_test.npy')
            num_file_path = os.path.join(file_dir, dataset +'_num_test.npy')
        all_data = np.load(data_file_path)
        all_num = np.load(num_file_path)
        self.all_data = torch.Tensor(all_data)
        self.all_num = torch.Tensor(all_num)
        
        past_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.OBS_TIME)
        future_frames = int(cfg.DATA.FREQUENCY * cfg.PREDICTION.PRED_TIME)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.traj_scale = cfg.PREDICTION.TRAJ_SCALE
        self.data_to_idx = {'eth':0, 'hotel':1, 'students003':2, 'zara01':3, 'zara02':4}

        # Setting variables to get all labels for a given frame and to convert to pixel coords
        homography_mat = dict()
        for data_name, _ in self.data_to_idx.items():
            h_path = os.path.join('datasets','eth_ucy','homography', f'H_{data_name}.txt')
            homography_mat[data_name] = np.loadtxt(h_path)
        label_file_names = {'eth':'biwi_eth.txt', 'hotel':'biwi_hotel.txt', 'students003':'students003.txt',
                                 'zara01': 'crowds_zara01.txt', 'zara02':'crowds_zara02.txt'}
        self.dfs = {}
        for data_name, file_name in label_file_names.items():
            labels_path = os.path.join('datasets', 'eth_ucy', 'labels')
            if data_name == 'students003':
                labels_path = os.path.join(labels_path, 'univ')
            else:
                labels_path = os.path.join(labels_path, data_name)
            labels_path = os.path.join(labels_path, file_name)
            labels_data = pd.read_csv(labels_path, sep = ' ',header=None)
            h_mat = homography_mat[data_name]
            frame_subset = labels_data[[13,15]].values
            labels_data[[17,18]] = world_to_pixel(frame_subset, h_mat, data_name, scale = 1).squeeze(axis = 1)
            self.dfs[data_name] = labels_data[[0, 1, 17, 18]].copy()
        self.idx_to_data = {v:k for k,v in self.data_to_idx.items()}
        self.img_transforms = img_transforms
        self.reshape_size = cfg.DATA.MIN_RESHAPE_SIZE
        self.mode = mode
        self.cfg = cfg

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self,item):
        all_seq = self.all_data[item] 
        all_seq[...,-2:] = all_seq[...,-2:] / self.traj_scale # Scaling only x,y world coords
        num = self.all_num[item]
        past_seq = all_seq[:,:self.past_frames, -2:]
        future_seq = all_seq[:,self.past_frames:, -2:]
        # Loading Images
        train_image_info = all_seq[:,:self.past_frames, :-2]
        train_image_info = train_image_info.reshape(-1, 4) # Flattening
        past_images = []
        pixel_coords = []
        traj_coords = []
        if self.cfg.MODEL.USE_PRETRAINED_UNET:
            for img_data in train_image_info:
                img_data = list(map(int, img_data.tolist()))
                frame_no = int(img_data[1])
                traj_coord = img_data[-2:]
                # If frames are not available then set a black frame
                if frame_no == -1:
                    img = torch.zeros((3, self.reshape_size, self.reshape_size))
                    pixel_coord = np.array([[-1,-1]])
                    traj_coord = [-1, -1]
                elif frame_no == 0 and traj_coord[0] == 0 and traj_coord[1] == 0:
                    img = torch.zeros((3,self.reshape_size,self.reshape_size))
                    pixel_coord = np.array([[-1,-1]])
                    traj_coord = [-1, -1]
                else:
                    data_name = self.idx_to_data[img_data[0]]
                    img_path = f'./datasets/eth_ucy/imgs/{data_name}/frame_{frame_no:04d}.jpg'

                    # Fetching Pixel coordinate for current frame
                    labels_data = self.dfs[data_name]
                    pixel_coord = labels_data[labels_data[0] == frame_no][[17,18]].values
                    img = plt.imread(img_path)

                    traj_coord = np.array([traj_coord])
                    pixel_coord = np.append(pixel_coord, traj_coord, axis = 0)

                    if self.img_transforms:
                        # Check if the pixel Coord is out of range then set it to -1, -1
                        # Since the agent is not properly visible in the frame
                        h, w = img.shape[:-1]
                        out_of_range_index = np.where(pixel_coord >= (h,w))[0]
                        pixel_coord[out_of_range_index,:] = (h-1, w-1)
                        transformed = self.img_transforms(image=img, keypoints=pixel_coord)
                        img = transformed['image']
                        pixel_coord = transformed['keypoints']
                        pixel_coord = np.array(pixel_coord)
                        if out_of_range_index.size != 0:
                            pixel_coord[out_of_range_index,:] = (-1, -1)

                        pixel_coord = pixel_coord.astype(np.int32)
                        traj_coord = pixel_coord[-1,:].tolist()

                img = img.unsqueeze(0).to(torch.float32)
                # Setting up binary map to indicate if there exists keypoint or not
                output_img_keypoint = torch.zeros(*img.shape[-2:])
                no_coords = np.where(pixel_coord != (-1,-1))[0]
                output_img_keypoint[pixel_coord[no_coords,0], pixel_coord[no_coords,1]] = 1
                output_img_keypoint = output_img_keypoint.unsqueeze(0)
                past_images.append(img)
                pixel_coords.append(output_img_keypoint)
                traj_coords.append(traj_coord)
            past_images = torch.cat(past_images)
            pixel_coords = torch.cat(pixel_coords)
            traj_coords = torch.tensor(traj_coords).reshape_as(past_seq)

        if self.mode == 'training' and self.cfg.DATA.SEQ_AUG:
            angle, translate_x, translate_y = 0, 0, 0
            # Getting Rotation
            prob_apply = random.uniform(0, 1)
            if prob_apply < float(self.cfg.DATA.SEQ_AUG_P):
                all_angles = list(range(0, 360))
                random.shuffle(all_angles)
                angle = all_angles[0]

            # Getting Translate X
            prob_apply = random.uniform(0, 1)
            if prob_apply < float(self.cfg.DATA.SEQ_AUG_P):
                all_translate_x = list(range(-10,10))
                random.shuffle(all_translate_x)
                translate_x = all_translate_x[0]
            
            # Getting Translate y
            prob_apply = random.uniform(0, 1)
            if prob_apply < float(self.cfg.DATA.SEQ_AUG_P):
                all_translate_y = list(range(-10,10))
                random.shuffle(all_translate_y)
                translate_y = all_translate_y[0]
            rotation_matrix = np.array([[np.cos(angle), np.sin(angle), translate_x],
                            [-np.sin(angle), np.cos(angle), translate_y],
                            [0, 0, 1]])
            
            for i in range(len(past_seq)):
                past_seq_pos = np.hstack((past_seq[i], np.ones((len(past_seq[i]),1))))
                fut_seq_pos = np.hstack((future_seq[i], np.ones((len(future_seq[i]),1))))
                past_seq_pos = np.dot(past_seq_pos, rotation_matrix.T)[:,:-1]
                fut_seq_pos = np.dot(fut_seq_pos, rotation_matrix.T)[:,:-1]
                past_seq[i] = torch.tensor(past_seq_pos)
                future_seq[i] = torch.tensor(fut_seq_pos)
        return past_seq, future_seq, num, past_images, pixel_coords, traj_coords

class data_generator(object):
    def __init__(self, dataset, past_frames, future_frames, 
                 data_root, traj_scale, split='train', 
                 phase='training'):
        self.past_frames = past_frames
        self.min_past_frames = past_frames
        self.future_frames = future_frames
        self.min_future_frames = future_frames
        self.frame_skip = 1
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if dataset in {'eth', 'hotel', 'univ', 'zara01', 'zara02'}:
            data_root = os.path.join(data_root, 'labels')          
            seq_train, seq_val, seq_test = get_ethucy_split(dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        log = 'log'
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        process_config = {}
        process_config['dataset'] = dataset
        process_config['past_frames'] = past_frames
        process_config['future_frames'] = future_frames
        process_config['frame_skip'] = self.frame_skip
        process_config['min_past_frames'] = past_frames
        process_config['min_future_frames'] = future_frames
        process_config['traj_scale'] = traj_scale
        
        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, process_config, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (self.min_past_frames - 1) * self.frame_skip - self.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        
        data = seq(frame)
        return data      

    def __call__(self):
        return self.next_sample()


class data_generator_new(object):
    def __init__(self, dataset, past_frames, future_frames, 
                 data_root, traj_scale, split='train', 
                 phase='training'):
        self.past_frames = past_frames
        self.min_past_frames = past_frames
        self.future_frames = future_frames
        self.min_future_frames = future_frames
        self.frame_skip = 1
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
            data_root = os.path.join(data_root,'data')          
            seq_train, seq_val, seq_test = get_ethucy_split(dataset)
            self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')

        process_func = preprocess
        self.data_root = data_root

        log = 'log'
        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'

        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        process_config = {}
        process_config['dataset'] = dataset
        process_config['past_frames'] = past_frames
        process_config['future_frames'] = future_frames
        process_config['frame_skip'] = self.frame_skip
        process_config['min_past_frames'] = past_frames
        process_config['min_future_frames'] = future_frames
        process_config['traj_scale'] = traj_scale
        

        for seq_name in self.sequence_to_load:
            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(data_root, seq_name, process_config, log, self.split, self.phase)

            num_seq_samples = preprocessor.num_fr - (self.min_past_frames - 1) * self.frame_skip - self.min_future_frames * self.frame_skip + 1
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor)
            
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        self.stack_size = 16
        self.max_scene_size = 8
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def shuffle(self):
        random.shuffle(self.sample_list)
        
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False

    def next_sample(self):
        cnt = 0
        seq_start_end = []
        all_loc = []
        all_loc_end = []
        while cnt < self.stack_size:
            if self.index >= self.num_total_samples:
                break
            sample_index = self.sample_list[self.index]
            seq_index, frame = self.get_seq_and_frame(sample_index)
            seq = self.sequence[seq_index]
            data = seq(frame)
            self.index += 1
            if data is not None:
                loc = torch.stack(data['pre_motion_3D'],dim=0)
                loc_end = torch.stack(data['fut_motion_3D'],dim=0)
                seq_start_end.append((cnt,cnt+loc.shape[0]))
                cnt += loc.shape[0]
                all_loc.append(loc)
                all_loc_end.append(loc_end)
        if len(all_loc) == 0:
            return None
        all_loc = torch.cat(all_loc,dim=0)
        all_loc_end = torch.cat(all_loc_end,dim=0)

        return all_loc, all_loc_end, seq_start_end

    def __call__(self):
        return self.next_sample()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--subset', type=str, default='eth',
                    help='Name of the subset.')
    parser.add_argument('--data_path', type=str, default = './datasets/eth_ucy')
    args = parser.parse_args()
    past_length = 8
    future_length = 12
    scale = 1

    generator_train = data_generator(args.subset, past_length, future_length, 
                                     args.data_path, scale, split='train', 
                                     phase='training')
    generator_test = data_generator(args.subset, past_length, future_length, 
                                    args.data_path, scale, split='test', 
                                    phase='testing')
    
    total_num = 5
    all_past_data = [] #(B,N,T,2)
    all_future_data = [] #(B,N,T,2)
    all_valid_num = []
    print('start process training data:')
    while not generator_train.is_epoch_end():
        data = generator_train()
        if data is not None:
            loc = torch.stack(data['pre_motion_3D'],dim=0)
            loc_end = torch.stack(data['fut_motion_3D'],dim=0)
            length = loc.shape[1]
            length_f = loc_end.shape[1]
            agent_num = loc.shape[0]
            loc = np.array(loc)
            loc_end = np.array(loc_end)
            if loc.shape[0] < total_num:
                for i in range(loc.shape[0]):
                    temp = np.zeros((total_num,length,2))
                    temp[0] = loc[i]
                    temp[1:agent_num] = np.delete(loc,i,axis=0)
                    all_past_data.append(temp[None])
                    
                    temp = np.zeros((total_num,length_f,2))
                    temp[0] = loc_end[i]
                    temp[1:agent_num] = np.delete(loc_end,i,axis=0)
                    all_future_data.append(temp)
                    all_valid_num.append(agent_num)
            else:
                for i in range(loc.shape[0]):
                    distance_i = np.linalg.norm(loc[:,-1] - loc[i:i+1,-1],dim=-1)
                    neighbors_idx = np.argsort(distance_i)
                    assert neighbors_idx[0] == i
                    neighbors_idx = neighbors_idx[:total_num]
                    temp = loc[neighbors_idx]
                    all_past_data.append(temp[None])
                    temp = loc_end[neighbors_idx]
                    all_future_data.append(temp[None])
                    all_valid_num.append(total_num)

    all_past_data = np.concatenate(all_past_data,dim=0)
    all_future_data = np.concatenate(all_future_data,dim=0)
    print(all_past_data.shape)
    print(all_future_data.shape)
    all_data = np.concatenate([all_past_data,all_future_data],dim=1)
    all_valid_num = np.array(all_valid_num)
    print(all_data.shape)
    print(all_valid_num.shape)
    np.save('processed_data/'+ args.subset +'_data_train.npy',all_data)
    np.save('processed_data/'+ args.subset +'_num_train.npy',all_valid_num)