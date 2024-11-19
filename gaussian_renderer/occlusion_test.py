
import torch
import numpy as np 
from torch import logical_or as tor
from scene.gaussian_model import GaussianModel
import time
import os

# occlusion test
class RayTracer:
    def __init__(self, pc:GaussianModel, res=128):
        self.splats = pc
        self.res = res
        self.voxel_grid = torch.zeros(res**3).bool().cuda()
    
    @torch.no_grad()
    def occlusion_test(self, rays_o, rays_d, update=True):
        res = self.res
        n = rays_o.shape[0]
        if update: # mark object voxels
            self.voxel_grid.zero_()
            splat_blocks = self.voxelize(self.splats.get_xyz)
            splat_idx = self.get_voxel_idx(splat_blocks)
            self.voxel_grid[splat_idx] = True
        
        # prepare 
        step = rays_d.sign() # [n][3]
        blocks = self.voxelize(rays_o) # [n][3]
        tDelta = (2.0/res) / (rays_d+1e-9) # [n][3]
        tHit = ((2.0/res)*(blocks+step)-rays_o) / (rays_d+1e-9)
        end = torch.zeros(n, dtype=bool).cuda()
        occ = torch.zeros(n, dtype=bool).cuda()

        iters = 0
        # ray voxel occlusion test
        while not torch.all(end):
            iters += 1
            # print(f"ITER {iters}: {torch.sum(end).item()} / {end.shape[0]}")
            argtMin = tHit.argmin(dim=1) # [n]
            for i in range(3):           # update blocks
                blocks[:,i] = torch.where(end, blocks[:,i], torch.where(
                    argtMin == i, blocks[:,i] + step[:,i], blocks[:,i]
                ))
            end = tor(end, self.get_voxel_oor(blocks))
            idx = self.get_voxel_idx(blocks)
            if iters > 0: occ = torch.where(end, occ, self.voxel_grid[idx]) # occlusion check
            end = tor(end, occ)
            for i in range(3):           # update tHit
                tHit[:,i] = torch.where(end, tHit[:,i], torch.where(
                    argtMin == i, tHit[:,i] + tDelta[:,i], tHit[:,i]
                ))

        # just check occlusion occur
        return occ # bool [n], where? [n]

    def voxelize(self, points): 
        points = (points + 1.0) / 2.0 * self.res # 0 ~ res
        points = points.floor()
        return points # [n][3]

    def get_voxel_oor(self, voxels): # out of range 
        res = self.res
        x_oor = tor(voxels[:,0]<0, voxels[:,0]>=res)
        y_oor = tor(voxels[:,1]<0, voxels[:,1]>=res)
        z_oor = tor(voxels[:,2]<0, voxels[:,2]>=res)
        return tor(x_oor, tor(y_oor, z_oor))

    def get_voxel_idx(self, voxels): # [n][3] -> [n]
        res = self.res
        idx = (res**2)*voxels[:,0] + res*voxels[:,1] + voxels[:,2]
        return idx.clamp(min=0,max=res**3-1).long()
        

