
import numpy as np
import torch
from networks.field import *

# generate cube environment maps
# class EnvLightModule

class EnvMapGenerator:
    def __init__(self, res=1024):
        self.resolution = res
        self.module = ViewOnlyNeRF().cuda()
        coords = cubemap_coordinates(res)
        angles = cartesian_to_spherical(coords)
        self.input = torch.tensor(angles).float().cuda()
        self.mipmap = []

    def get_maps(self, update=False, levels=5):
        if update:
            output = self.module.forward(self.input)
            res = self.resolution
            self.mipmap = [output.reshape(6, res, res, 3)]
            print("ENVMAP: ", self.mipmap[0].min().item(), self.mipmap[0].max().item(), self.mipmap[0].mean().item())
            for i in range(1,levels):
                self.extend_mipmap()
        return self.mipmap
        
    def extend_mipmap(self):
        map_prev = self.mipmap[-1] # [6][res][res][3]
        xx = map_prev[:, ::2, ::2, :]
        xy = map_prev[:, ::2, 1::2, :]
        yx = map_prev[:, 1::2, ::2, :]
        yy = map_prev[:, 1::2, 1::2, :]
        map_curr = (xx + xy + yx + yy) / 4.0
        self.mipmap.append(map_curr)
    
    def get_direct(self, rays_o, rays_d, rays_om):
        n = rays_o.shape[0]
        tHit = (rays_d.sign()-rays_o) / (rays_d+1e-9) # [n][3]
        tMin = tHit.min(dim=1)
        rays_t, argtMin = tMin.values, tMin.indices # [n]
        rays_end = rays_o + rays_t[:,None].clamp(0,4) * rays_d # [n][3]

        face = torch.empty((n,1)).long().cuda() # [n][1]
        for i in range(3): # select faces 0~5
            face[argtMin==i] = 2*i + (rays_d[argtMin==i,i] < 0)[:,None]
        coord = torch.where( # [n][2]
            face < 2, rays_end[:,[0,1]],
            torch.where(
                face < 4, rays_end[:,[0,2]],
                rays_end[:,[1,2]]
            )
        ).clamp(-1.0,1.0-1e-6)

        # compute mip_level
        mip_level = torch.zeros(n)
        # TODO
        
        # for now, colors from grid without interpolation
        grid = ((coord+1)*(float(self.resolution)/(2.0**(mip_level[:,None]+1)))).long() # [n][2]
        colors = torch.empty((n,3)).cuda()
        if not torch.all(mip_level < len(self.mipmap)): 
            return NotImplementedError
        
        for i in range(len(self.mipmap)):
            mip = self.mipmap[i]
            fc = face[mip_level==i,0]
            x, y = grid[mip_level==i,0], grid[mip_level==i,1]
            if fc.shape[0] > 0:
                # print(fc.max().item(), fc.min().item())
                # print(x.max().item(), x.min().item())
                # print(y.max().item(), y.min().item())
                colors[mip_level==i] = mip[fc,x,y]
        
        return colors 


# mathematical functions
def cubemap_coordinates(res):
    units = np.linspace(-1+1.0/res, 1-1.0/res, res, endpoint=True)
    grid1, grid2 = np.meshgrid(units, units)
    grid1, grid2 = grid1.flatten(), grid2.flatten()
    p, m = np.ones_like(grid1), -np.ones_like(grid1)
    faces = []
    faces.append(np.stack([p, grid1, grid2], axis=-1)) 
    faces.append(np.stack([m, grid1, grid2], axis=-1))
    faces.append(np.stack([grid1, p, grid2], axis=-1))
    faces.append(np.stack([grid1, m, grid2], axis=-1))
    faces.append(np.stack([grid1, grid2, p], axis=-1))
    faces.append(np.stack([grid1, grid2, m], axis=-1))
    return np.vstack(faces)

def cartesian_to_spherical(xyz):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x**2+y**2+z**2) # radius
    theta = np.arccos(z/r) # polar angle (0 to pi)
    phi = np.arctan2(y,x) # azimuthal angle (-pi to pi)
    return np.stack([theta, phi], axis=-1) 
