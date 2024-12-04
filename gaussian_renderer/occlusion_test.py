
import torch
import numpy as np 
from torch import logical_or as tor
from torch.nn.functional import one_hot
from scene.gaussian_model import GaussianModel
import ray
import time
import os

# occlusion test
@ray.remote(num_gpus=1)
class RayTracer:
    def __init__(self, res=128):
        self.res = res
        print("ray.get_gpu_ids():", ray.get_gpu_ids())
        gpu_id = ray.get_gpu_ids()[0]
        self.device = torch.device(f'cuda:{gpu_id}')
        print("DEVICE", self.device)
        self.voxel_grid = torch.zeros(res**3).bool().cuda()
    
    @torch.no_grad()
    def occlusion_test(self, rays_o, rays_d, xyz, update=True):
        start = time.time()
        res = self.res
        rays_o = torch.tensor(rays_o).cuda()
        rays_d = torch.tensor(rays_d).cuda()
        xyz = torch.tensor(xyz).cuda()
        resv = torch.tensor([self.res**2, self.res, 1.0]).cuda()

        n = rays_o.shape[0]
        if update: # mark object voxels
            self.voxel_grid.zero_()
            splat_blocks = self.voxelize(xyz)
            splat_idx = (splat_blocks @ resv).clamp(0, self.res**3-1).long() 
            self.voxel_grid[splat_idx] = True
        
        # prepare 
        step = rays_d.sign() # [n][3]
        blocks = self.voxelize(rays_o) # [n][3]
        tDelta = (2.0/res) / (rays_d+1e-9) # [n][3]
        tHit = ((2.0/res)*(blocks+step)-rays_o) / (rays_d+1e-9)
        
        end = torch.zeros(n, dtype=bool).cuda()
        temp = torch.zeros(n, dtype=bool).cuda()
        occ = torch.zeros(n, dtype=bool).cuda()

        iters = 0
        # ray voxel occlusion test
        while not torch.all(end):
            iters += 1
            tempidx = (~temp).nonzero().squeeze()
            tHit = tHit.index_select(0, tempidx)
            tDelta = tDelta.index_select(0, tempidx)
            blocks = blocks.index_select(0, tempidx)
            step = step.index_select(0, tempidx)

            argtMin = tHit.argmin(dim=1) # [n]
            add = one_hot(argtMin, num_classes=3).to(blocks.dtype)
            blocks += step*add 
            tHit += tDelta*add

            temp = end.clone() # hold end tensor before changes
            end[~temp] = ((blocks < 0)|(blocks >= self.res)).any(dim=1) 
            idx = (blocks @ resv).clamp(0, self.res**3-1).long() 
            if iters > 5: occ[~temp] = self.voxel_grid[idx] # occlusion check
            end = tor(end, occ)
            temp = end.index_select(0, (~temp).nonzero().squeeze()) # temp = end[~temp]


        # just check occlusion occur
        end = time.time()
        print("PROCESS:", end-start)
        return occ # bool [n], where? [n]

    def voxelize(self, points): 
        points = (points + 1.0) / 2.0 * self.res # 0 ~ res
        points = points.floor()
        return points # [n][3]
        

