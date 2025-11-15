# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import joblib
import random
from pacer.utils.flags import flags
from pacer.env.tasks.base_task import PORT, SERVER
import time
Dire = [0, 1, 2, 3]
move1 = [0,0,1,-1]
move2 = [-1,1,0,0]  
theta = [np.pi/2,-np.pi/2,0,-np.pi]
class TrajGenerator():
    def __init__(self, num_envs, episode_dur, num_verts, device, dtheta_max,
                 speed_min, speed_max, accel_max, sharp_turn_prob,terrain,style_idx):

        self.num_envs = num_envs
        self._device = device
        self._dt = episode_dur / (num_verts - 1)
        self._dtheta_max = dtheta_max
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._accel_max = accel_max
        self._sharp_turn_prob = sharp_turn_prob

        self._verts_flat = torch.zeros((num_envs * num_verts, 3), dtype=torch.float32, device=self._device)
        self._verts = self._verts_flat.view((num_envs, num_verts, 3))

        env_ids = torch.arange(self.get_num_envs(), dtype=np.int)
        self.heading = torch.zeros(num_envs, 1)
        self.terrain = terrain
        self.style_idx = style_idx

        ## Generate speed coefficient
        # self.velocity_scales = None
        # self.style_speed_ranges = {
        #-1: (0.9, 1.3), # without stylization
        #0: (0.7,1.0), # BentForward, bending forward 40-70 years old
        #1: (0.8,1.2), # BentKnees, knee flexion 30-60 years old
        #2: (1.0, 1.3), # BigSteps, take big strides for ages 20-50
        #3: (0.6, 0.9), # Crouched, squatting and walking 30-50 years old
        #4: (0.8,1.2), # CrowdAvoidance, avoid crowds aged 20-60
        #5: (0.6, 0.9), # Drunk, intoxicated 20-50 years old
        #6: (0.9, 1.2), # Followed, followed by 20-50 years old
        #7: (0.8,1.0), # HandsInPockets, Hand Pocket 20-40 years old
        #8: (0.6, 0.9), # Heavyset, heavy build, walking age 30-70 years old
        #9: (0.8, 1.2), # KarateChop, Karate 20-40 years old
        #10: (0.8,1.2), # Kick, Kick and Walk 20-40 Years Old
        #11: (0.8,1.0), # LeanLeft, left leaning 20-50 years old
        #12: (0.8,1.0), # LeanRight, right leaning 20-50 years old
        #13: (0.8, 1.2), # LegsApart, walking with split legs for ages 20-50
        #14: (0.6, 0.8), # LimpLeft, left leg limp 40-70 years old
        #15: (0.6, 0.8), # LimpRight, right leg limp 40-70 years old
        #16: (0.9, 1.2), # LookUp, looking up and walking 20-50 years old
        #17: (1.0, 1.3), # Lunge, taking a step towards 20-40 years old
        #18: (0.9,1.4), # Neutral, normal gait 20-60 years old
        #19: (0.6, 0.9), # Old, staggering 50-80 years old
        #20: (0.8,1.0), # OnHeels, heel to ground age 20-50
        #21: (0.8,1.2), # PigeonToed, with a Chinese zodiac sign between 10-40 years old
        #22: (1.2,1.3), # Rushed, hurried 20-50 years old
        #23: (0.8,1.0), # SlideFeet, age 20-50
        #24: (0.6, 1.2), # StartStop, stop and walk between 20-60 years old
        #25: (1.0, 1.2), Strutting, high toed 20-40 years old
        #26: (0.6, 0.9), # WalkingDickLeft, left hand leaning on crutches for ages 50-80
        #27: (0.6, 0.9), # WalkingDickRight, using a cane in the right hand for ages 50-80
        #28: (0.9, 1.2), # WhirlArms, swinging arms 20-50 years old
        #29: (0.9, 1.2), # WiggleHips, wriggling hips 20-40 years old
        # }
        # self._init_velocity_scales()
        return
    

              

    def set_trajectory(self, env_ids, waypoints):
        """
        Args:
            waypoints: (n, num_points, 3)
        """
        n = len(env_ids)
        num_verts = self.get_num_verts()
        
        # Calculate target direction and distance
        init_pos = waypoints[:, 0] 
        target_pos = waypoints[:, -1]
        total_direction = target_pos - init_pos
        
        speed = torch.ones([n, num_verts - 1], device=self._device) * 3  # Set the speed to 3
        dtheta = torch.zeros_like(speed)
        
        thetas = torch.atan2(total_direction[:, 1], total_direction[:, 0])
        dtheta[:, 0] = thetas
        dtheta = torch.cumsum(dtheta, dim=-1)
        
        seg_len = speed * self._dt
        dpos = torch.stack([torch.cos(dtheta), torch.sin(dtheta), torch.zeros_like(dtheta)], dim=-1)
        dpos *= seg_len.unsqueeze(-1)
        
        # Calculate all trajectory points
        all_points = torch.zeros((n, num_verts, 3), device=self._device)
        all_points[:, 0] = init_pos
        current_directions = dtheta[:, 0].clone()
        # When the direction is invalid, increase the angle offset and try a new direction
        angle_offsets = torch.tensor([-np.pi/4, np.pi/4, -np.pi/2, np.pi/2], device=self._device)

        for i in range(1, num_verts):
            # Pre calculate the next point
            next_points = all_points[:, i-1] + dpos[:, i-1]
            
            # Check if the checkpoint is within navmesh
            points_to_check = next_points[:, None, :2]  # [n, 1, 2]
            valid_points = self.terrain.inside_navmesh(points_to_check)
            
            # If all points are valid, update directly
            if valid_points.all():
                all_points[:, i] = next_points
                continue
            
            # For invalid points, try to find new effective directions
            invalid_mask = ~valid_points.squeeze()
            if invalid_mask.any():
                found_valid_direction = torch.zeros_like(invalid_mask, dtype=torch.bool)
                
                for offset in angle_offsets:
                    still_invalid = invalid_mask & ~found_valid_direction
                    if not still_invalid.any():
                        break
                        
                    # Try a new direction
                    new_direction = current_directions[still_invalid] + offset
                    new_dpos = torch.stack([
                        torch.cos(new_direction), 
                        torch.sin(new_direction),
                        torch.zeros_like(new_direction)
                    ], dim=-1) * seg_len[still_invalid, i-1].unsqueeze(-1)
                    
                    test_points = all_points[still_invalid, i-1] + new_dpos

                    test_points_reshaped = test_points[:, None, :2]
                    test_valid = self.terrain.inside_navmesh(test_points_reshaped)
                    
                    # Update points that find effective directions
                    valid_indices = still_invalid.nonzero().squeeze()[test_valid.squeeze()]
                    if len(valid_indices.shape) == 0:
                        valid_indices = valid_indices.unsqueeze(0)
                    
                    if valid_indices.numel() > 0:
                        current_directions[valid_indices] = new_direction[test_valid.squeeze()]
                        next_points[valid_indices] = test_points[test_valid.squeeze()]
                        found_valid_direction[valid_indices] = True

                dtheta[:, i-1] = current_directions
            
            all_points[:, i] = next_points
        
        self._verts[env_ids] = all_points
        
        return self._verts[env_ids]

    def reset(self, env_ids, init_pos,last_pos, mass, delta_t):
        n = len(env_ids)
        set_traj = False
        if (n > 0):
            if set_traj:
                target_points = torch.tensor([-40,0,0], device=self._device).repeat(n, 1)
                waypoints = torch.stack([
                    init_pos,
                    target_points
                ], dim=1)  # shape: (n, 2, 3)
                traj = self.set_trajectory(env_ids, waypoints)
            else:
                num_verts = self.get_num_verts()
                
                speed = torch.ones([n, num_verts - 1],device=self._device)
                dtheta = torch.zeros_like(speed)

                ##############social force
                speed_now = (last_pos-init_pos)/delta_t
                if last_pos[0,0] == float("inf"):
                    speed_now = torch.zeros_like(speed_now)
                mass = [1]*n

                targets_next,velocity = self.terrain.social_force_simulate(
                    init_pos,
                    speed_now[:,:2],
                    mass,
                    self._dt,
                    self.style_idx[env_ids]
                    )


                # TODO
                # Optimize social force calculation

                thetas = torch.atan2(velocity[:,1],velocity[:,0])
                dtheta[:,0] = thetas
                velocity_norm = torch.norm(velocity,dim=1)
                speed = speed * velocity_norm.unsqueeze(1)
                ###########################
                
                

                dtheta = torch.cumsum(dtheta, dim=-1)
                
                seg_len = speed * self._dt

                dpos = torch.stack([torch.cos(dtheta), torch.sin(dtheta), torch.zeros_like(dtheta)], dim=-1)
                dpos *= seg_len.unsqueeze(-1)
                dpos[..., 0, 0:2] += init_pos[..., 0:2]
                vert_pos = torch.cumsum(dpos, dim=-2)

                self._verts[env_ids, 0, 0:2] = init_pos[..., 0:2]
                self._verts[env_ids, 1:] = vert_pos
                if flags.real_path:
                    rids = random.sample(self.traj_data.keys(), n)
                    traj = torch.stack([
                        torch.from_numpy(
                            self.traj_data[id]['coord_dense'])[:num_verts]
                        for id in rids
                    ],dim=0).to(self._device).float()

                    traj[..., 0:2] = traj[..., 0:2] - (traj[..., 0, 0:2] - init_pos[..., 0:2])[:, None]
                    self._verts[env_ids] = traj
        return
    
    def input_new_trajs(self, env_ids):
        import json
        import requests
        from scipy.interpolate import interp1d
        x = requests.get(
            f'http://{SERVER}:{PORT}/path?num_envs={len(env_ids)}')

        data_lists = [value for idx, value in x.json().items()]
        coord = np.array(data_lists)
        x = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1])
        fx = interp1d(x, coord[..., 0], kind='linear')
        fy = interp1d(x, coord[..., 1], kind='linear')
        x4 = np.linspace(0, coord.shape[1] - 1, num = coord.shape[1] * 10)
        coord_dense = np.stack([fx(x4), fy(x4), np.zeros([len(env_ids), x4.shape[0]])], axis = -1)
        coord_dense = np.concatenate([coord_dense, coord_dense[..., -1:, :]], axis = -2)
        self._verts[env_ids] = torch.from_numpy(coord_dense).float().to(env_ids.device)
        return self._verts[env_ids]


    def get_num_verts(self):
        return self._verts.shape[1]

    def get_num_segs(self):
        return self.get_num_verts() - 1

    def get_num_envs(self):
        return self._verts.shape[0]

    def get_traj_duration(self):
        num_verts = self.get_num_verts()
        dur = num_verts * self._dt
        return  dur

    def get_traj_verts(self, traj_id):
        return self._verts[traj_id]

    def calc_pos(self, traj_ids, times):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        return pos

    def mock_calc_pos(self, env_ids, traj_ids, times, query_value_gradient):
        traj_dur = self.get_traj_duration()
        num_verts = self.get_num_verts()
        num_segs = self.get_num_segs()

        traj_phase = torch.clip(times / traj_dur, 0.0, 1.0)
        seg_idx = traj_phase * num_segs
        seg_id0 = torch.floor(seg_idx).long()
        seg_id1 = torch.ceil(seg_idx).long()
        lerp = seg_idx - seg_id0

        pos0 = self._verts_flat[traj_ids * num_verts + seg_id0]
        pos1 = self._verts_flat[traj_ids * num_verts + seg_id1]

        lerp = lerp.unsqueeze(-1)
        pos = (1.0 - lerp) * pos0 + lerp * pos1

        new_obs, func = query_value_gradient(env_ids, pos)
        if not new_obs is None:
            # ZL: computes grad
            with torch.enable_grad():
                new_obs.requires_grad_(True)
                new_val = func(new_obs)

                disc_grad = torch.autograd.grad(
                    new_val,
                    new_obs,
                    grad_outputs=torch.ones_like(new_val),
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True)

        return pos
