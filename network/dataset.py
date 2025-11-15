# This code is based on https://github.com/AIGAnimation/CAMDM
import sys
sys.path.append('./')

import torch
import pickle
import random
import numpy as np

import utils.nn_transforms as nn_transforms
from scipy.ndimage import gaussian_filter1d

from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


MATCH_OFFSET = {'BR':0,'BW':2571,'FR':6498,'FW':8433,'SW':11992,'SR':13355}

class MotionDataset(Dataset):
    '''
    rot_req: str, rotation format, 'q'|'6d'|'euler'
    window_size: int, window size for each clip
    '''
    def __init__(self, pkl_path,matching_pkl_path,source_data_path, rot_req, offset_frame, dtype=np.float32, limited_num=-1):
        self.pkl_path, self.rot_req,self.dtype = pkl_path, rot_req, dtype
        self.matching_pkl_path = matching_pkl_path
        self.source_data_path = source_data_path
        self.rotations_list, self.root_pos_list = [], []
        self.local_conds = []
        self.global_conds = {'style': []}
        self.matching_idx = []
        self.rot_feat_dim = {'q': 4, '6d': 6, 'euler': 3}
        
        data_source = pickle.load(open(pkl_path, 'rb'))
        matching_list = pickle.load(open(matching_pkl_path, 'rb'))
        self.T_pose = data_source['T_pose']

        for motion_item in tqdm(data_source['motions']):
            frame_num = motion_item['local_joint_rotations'].shape[0]
            if motion_item['style'] == 'Neutral':
                for frame_idx in range(frame_num):
                    self.local_conds.append(motion_item['local_joint_rotations'][frame_idx].astype(dtype))
            
            style = motion_item['style']
            filepath = motion_item['filepath'].split("/")[-1]
            action = filepath.split("_")[-1][:2]
            if action == 'ID' or action == 'TR':
                    continue
            match_idxs = matching_list[filepath]
            
            for frame_idx in range(frame_num):
                
                rotation = motion_item['local_joint_rotations'][frame_idx].astype(dtype)
                self.rotations_list.append(rotation)

                self.global_conds['style'].append(style)
                self.matching_idx.append(MATCH_OFFSET[action]+match_idxs[frame_idx])


        self.joint_num, self.per_rot_feat = self.rotations_list[0].shape[-2], self.rot_feat_dim[rot_req]
        self.mask = np.ones(1, dtype=bool)
        self.style_set = sorted(list(set(self.global_conds['style'])))
        
        
        print('Dataset loaded, trained with %d style frames, %d source frames in total' % (len(self.rotations_list), len(self.local_conds)))
        
        
    def __len__(self):
        return len(self.rotations_list)
    
    def __getitem__(self, idx):
        rotations = self.rotations_list[idx].copy()#[njoints,nfeats]
        root_rot = rotations[0]
        root_rotation_xyzw = root_rot[[1, 2, 3, 0]]#[bs,njoin]
        
        
        theta = np.random.uniform(0, 2*np.pi)
        rot_vec = R.from_rotvec(np.array([0,theta,0]))
        root_rotation = (rot_vec*R.from_quat(root_rotation_xyzw)).as_quat()[..., [3, 0, 1, 2]]
        rotations[0] = root_rotation
        rotations = nn_transforms.get_rotation(rotations.astype(self.dtype), self.rot_req).unsqueeze(0)
        
        source_pos = self.local_conds[self.matching_idx[idx]] 
        source_pos[0,:] = root_rotation
        source_pos = nn_transforms.get_rotation(source_pos.astype(self.dtype), self.rot_req).unsqueeze(0)

        style_idx = int(self.style_set.index(self.global_conds['style'][idx]))
        
        return {
            'data': rotations,
            'conditions': {
                'source_pos': source_pos,
                'style': self.global_conds['style'][idx],
                'style_idx': style_idx,
                'mask': self.mask
            }
        }     



# Case test for the dataset class
if __name__ == '__main__':
    
    import time
    import torch
    
    pkl_path = 'data/pkls/100style_filtered.pkl'
    rot_req = '6d'
    train_data = MotionDataset(pkl_path, rot_req, offset_frame=1, dtype=np.float32)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=8, drop_last=False, pin_memory=True)
    
    do_export_test = False
    do_loop_test = True
    
    if do_export_test:
        T_pose = train_data.T_pose
        data = train_dataloader.__iter__().__next__()
        rotations = nn_transforms.repr6d2quat(data['rotations']).numpy()
        root_pos = data['root_pos'].numpy()
         
        for i in range(10):
            T_pose_template = T_pose.copy()
            T_pose_template.rotations = rotations[i]
            T_pose_template.positions = np.zeros((rotations[i].shape[0], T_pose_template.positions.shape[1], T_pose_template.positions.shape[2]))
            T_pose_template.positions[:, 0] = root_pos[i]
            T_pose_template.export('save/visualization/example_bvh/%s.bvh' % i, save_ori_scal=True)  
             
    if do_loop_test:
        times = []
        start_time = time.time()
        for data in train_dataloader:
            end_time = time.time()
            print('Data loading time for each iteration: %ss' % (end_time - start_time))
            times.append(end_time - start_time)
            start_time = end_time
        avg_time = sum(times) / len(times)
        print('Average data loading time: %ss' % avg_time)
        print('Entire data loading time: %ss' % sum(times))
    
