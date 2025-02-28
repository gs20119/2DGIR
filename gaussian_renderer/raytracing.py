
import torch
import abc
import numpy as np
from torch.optim import Adam
from gaussian_renderer.occlusion_test import RayTracer
from scene.gaussian_model import GaussianModel
from networks.envmaps import EnvMapGenerator
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
from arguments import ModelParams
import ray, os, time
import math

# Compute BRDF Rendering Equation (RayTracing + Importance Sampling)
class BRDFRenderer:
    def __init__(self, pc:GaussianModel, args:ModelParams):
        # later, we will get these constants from cfg
        self.Sd = args.rays_diffuse
        self.Ss = args.rays_specular
        self.S = self.Sd + self.Ss # of incident rays for a point
        self.splats = pc
        self.steps = 1
        self.sampler = Sampler(pc, self.Sd, self.Ss) # ray sampler
        self.env = EnvMapGenerator() # environment lightmap
        self.lr_manager = WarmUpCosLR({
            "lr_init": 2.0e-4,
            "lr_step": 100000,
            "lr_rate": 0.5,
        })
        self.optimizer = self.lr_manager.construct_optimizer(Adam, self.env.module)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        ncpus, ngpus = 20, torch.cuda.device_count()
        ray.init(num_cpus=ncpus, num_gpus=ngpus) 
        self.tracers = [RayTracer.remote() for _ in range(ngpus)] 
        self.ngpus = ngpus

    
    # return outgoing radiance along given rays
    def color_recursive(self, rays_from, rays_d, steps=1): 
        # rays_from: gaussian indices [n]
        # rays_d: directions [n][3] 
        if steps == 0: # terminate recursion
            return None # return SH color of gaussians
        
        # sample incident rays (direction)
        inrays_diff, coef_diff, cone_diff = self.sampler.sample_diffuse(rays_from, rays_d)
        inrays_spec, coef_spec, cone_spec = self.sampler.sample_specular(rays_from, rays_d)
        incid_rays = torch.concatenate((inrays_diff, inrays_spec), dim=1) # [n][S][3]
        solid_angles = torch.concatenate((cone_diff, cone_spec), dim=1) # [n][S]

        # occlusion test
        incid_rays = incid_rays.reshape(-1,3) # [n*S][3]
        solid_angles = solid_angles.reshape(-1) # [n*S]
        rays_o = self.splats.get_xyz[rays_from].repeat_interleave(self.S,dim=0) # [n*S][3]
        
        st = time.time()
        ro_split = split_tensor_gpus(rays_o, self.ngpus)
        rd_split = split_tensor_gpus(incid_rays, self.ngpus)
        xyz = self.splats.get_xyz.cpu().detach().numpy()
        end = time.time()

        remotes = [
            tracer.occlusion_test.remote(ro_split[i-1], rd_split[i-1], xyz) 
            for i, tracer in enumerate(self.tracers)
        ]

        st = time.time()
        results = ray.get(remotes)
        end = time.time()
        print("OCC_TEST:", end-st)

        occ = gather_tensors_gpus(results)

        # get direct light (occ == 0)
        self.env.get_maps(update=True)
        direct_lights = self.env.get_direct(rays_o[occ==0], incid_rays[occ==0], solid_angles[occ==0])

        # get indirect light (occ == 1)
        # indirect_lights = self.color_recursive(hit_to[occ==1], -incid_rays[occ==1], steps-1) 
        # just get SH color of gaussians for now 
        shs = self.splats.get_ind_features[rays_from]
        shs = shs.transpose(1,2).view(-1,3,(self.splats.max_sh_degree+1)**2) # [n][3][16]
        count = occ.view(-1,self.S).sum(dim=1)
        shs = shs.repeat_interleave(count, dim=0) # [ind][3][16]
        sh2rgb = eval_sh(self.splats.active_sh_degree, shs, incid_rays[occ==1])
        indirect_lights = torch.clamp_min(sh2rgb+0.5, 1e-9) 

        # compute integrals
        incid_lights = torch.zeros_like(incid_rays).cuda() # [n*S][3]
        incid_lights[~occ] = direct_lights
        incid_lights[occ] = indirect_lights
        incid_lights = incid_lights.reshape(-1, self.S, 3) # [n][S][3]

        color_diff = (incid_lights[:,:self.Sd,:]*coef_diff).mean(dim=1)
        color_spec = (incid_lights[:,self.Sd:,:]*coef_spec).mean(dim=1)
        out_lights = color_diff + color_spec # [n][3]

        return out_lights


def split_tensor_gpus(x, num_gpus, dim=0):
    chunks = torch.chunk(x, num_gpus, dim=dim)
    return [chunk.cpu().detach().numpy() for i, chunk in enumerate(chunks)]

def gather_tensors_gpus(chunks, dim=0):
    xs = [chunk.cuda(0) for chunk in chunks]
    return torch.cat(xs, dim=dim)


# sampling incident rays
class Sampler:
    def __init__(self, pc:GaussianModel, Sd, Ss):
        self.splats = pc # requires their normal, roughness
        self.Xid = torch.tensor([[i/Sd, van_der_corput_bitwise(i)] for i in range(Sd)]).cuda()
        self.Xis = torch.tensor([[i/Ss, van_der_corput_bitwise(i)] for i in range(Ss)]).cuda()
    
    @torch.no_grad
    def importanceSampleCosine(self, Xi, rotate): 
        theta = 2.0 * math.pi * Xi[:,0] # [S]
        sinPhi = torch.sqrt(Xi[:,1])
        cosPhi = torch.sqrt(1.0-sinPhi**2) # [S]
        samples = torch.stack([
            torch.cos(theta)*sinPhi, 
            torch.sin(theta)*sinPhi, cosPhi], dim=1) # [S][3]
        samples = torch.matmul(rotate[:,None,:,:], samples[None,:,:,None]).squeeze(-1)
        return samples # [n][S][3]

    def sample_diffuse(self, rays_from, rays_d): # importance sampling from cos-weight
        n = rays_from.shape[0]
        S = self.Xid.shape[0]
        
        # get materials
        material = self.splats.get_material[rays_from]
        base, rough, metal = material[:,:3], material[:,3], material[:,4]
        
        # compute vectors  
        view = rays_d[:,None,:]
        normal = self.splats.get_normal[rays_from][:,None,:]          # [n][1][3]
        rotate = build_rotation(self.splats.get_rotation[rays_from])

        light = self.importanceSampleCosine(self.Xid, rotate)         # [n][S][3]
        NoL = (normal @ light.transpose(-1,-2)).transpose(-1,-2)      # [n][S][1]

        # compute coefficients
        coef = (1.0-metal[:,None])*base                               # [n][3]
        coef = coef[:,None,:]                                         # [n][1][3]
        pdf = NoL / math.pi                                           # [n][S][1]

        # compute solid angles
        cone_angles = (1.0 / (S*pdf)).squeeze()                       # [n][S]

        return light, coef, cone_angles # [n][S][3], [n][1][3], [n][S]
    
    # From Unreal Engine 4
    @torch.no_grad
    def importanceSampleGGX(self, Xi, roughness, rotate):
        a = roughness[:,None] ** 2 # [n][1]
        theta = 2.0 * math.pi * Xi[:,0] # phi = 0~pi/2 uniform sampling [S]
        cosPhi = torch.sqrt((1.0-Xi[None,:,1]) / (1.0+(a**2-1.0)*Xi[None,:,1])) # theta = 2pi GGX sampling [n][S]
        sinPhi = torch.sqrt(1.0-cosPhi**2) # [n][S]
        samples = torch.stack([
            torch.cos(theta)*sinPhi, 
            torch.sin(theta)*sinPhi, cosPhi], dim=2) # [n][S][3]
        s = Xi.shape[0]
        samples = (rotate.unsqueeze(1) @ samples.unsqueeze(-1)).squeeze(-1)  # [n][3][3] x [n][S][3]
        return samples #[n][S][3]

    def GGX(self, NoH):
        a2 = roughness ** 4
        a2 = a2[:,None,None] # [n][1][1]
        return a2 / (math.pi * ((NoH**2)*(a2-1)+1)**2) # [n][S][1]
        
    # Smith term for GGX
    def gsmith(self, roughness, NoV, NoL):
        a2 = roughness ** 4
        a2 = a2[:,None,None] # [n][1][1]
        G_SmithV = NoV + torch.sqrt((1.0-a2)*(NoV**2) + a2)
        G_SmithL = NoL + torch.sqrt((1.0-a2)*(NoL**2) + a2)
        return (4*NoV*NoL) / (G_SmithV*G_SmithL)
    
    def sample_specular(self, rays_from, rays_d): # importance sampling from GGX
        n = rays_d.shape[0]
        S = self.Xis.shape[0]
        
        # get materials
        material = self.splats.get_material[rays_from]
        base, rough, metal = material[:,:3], material[:,3], material[:,4]
        
        # compute vectors 
        view = rays_d[:,None,:]                                       # [n][1][3]
        normal = self.splats.get_normal[rays_from][:,None,:]          # [n][1][3]
        rotate = build_rotation(self.splats.get_rotation[rays_from])  # [n][3][3]
        NoV = (normal @ view.transpose(-1,-2))+1e-6                   # [n][1][1]

        half = self.importanceSampleGGX(self.Xis, rough, rotate)      # [n][S][3]
        VoH = (view @ half.transpose(-1,-2)).transpose(-1,-2)         # [n][S][1]
        light = 2*VoH*half-view                                       # [n][S][3]
        NoL = (normal @ light.transpose(-1,-2)).transpose(-1,-2)      # [n][S][1]
        NoH = (normal @ half.transpose(-1,-2)).transpose(-1,-2)       # [n][S][1]
        mask = torch.logical_or(NoL<=0, NoV<=0)                       # [n][S][1]
        #assert torch.all(NoV>=0)

        # compute coefficients (BRDF/pdf)
        NoV = NoV.clamp(1e-6,1)
        NoL = NoL.clamp(1e-6,1)
        NoH = NoH.clamp(1e-6,1)
        VoH = VoH.clamp(1e-6,1)
        D = self.GGX(NoH)                                             # [n][S][1]
        G = self.gsmith(rough, NoV, NoL)                              # [n][S][1]
        F0 = 0.04*(1-metal[:,None]) + metal[:,None]*base              # [n][3]
        F0 = F0[:,None,:]                                             # [n][1][3]
        Fc = (1.0-VoH)**5                                             # [n][S][1]
        F = F0 + (1.0-F0)*Fc                                          # [n][S][3]
        coef = F*G*VoH / (NoH*NoV).clamp(1e-6,1)                      # [n][S][3]
        coef = torch.where(mask, 0.0, coef)
        pdf = D*NoH / (4.0*VoH)                                       # [n][S][1]

        # compute solid angles
        cone_angles = (1.0 / (S*pdf)).squeeze()                       # [n][S]

        return light, coef, cone_angles # [n][S][3], [n][S][3], [n][S]


# util funtions
def van_der_corput_bitwise(bits: int) -> float:
    bits = bits & 0xFFFFFFFF
    bits = ((bits << 16) | (bits >> 16)) & 0xFFFFFFFF
    bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
    bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
    bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF   
    bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
    return float(bits) * 2.3283064365386963e-10

# From NeRO 
class LearningRateManager(abc.ABC):
    @staticmethod
    def set_lr_for_all(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def construct_optimizer(self, optimizer, network):
        # may specify different lr for different parts
        # use group to set learning rate
        paras = network.parameters()
        return optimizer(paras, lr=1e-3)

    @abc.abstractmethod
    def __call__(self, optimizer, step, *args, **kwargs):
        pass

class WarmUpCosLR(LearningRateManager):
    default_cfg={
        'end_warm': 5000,
        'end_iter': 300000,
        'lr': 5e-4,
    }
    def __init__(self, cfg):
        cfg={**self.default_cfg,**cfg}
        self.warm_up_end = cfg['end_warm']
        self.learning_rate_alpha = 0.05
        self.end_iter = cfg['end_iter']
        self.learning_rate = cfg['lr']

    def __call__(self, optimizer, step, *args, **kwargs):
        if step < self.warm_up_end:
            learning_factor = step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        lr = self.learning_rate * learning_factor
        self.set_lr_for_all(optimizer, lr)
        return lr

name2lr_manager={
    'warm_up_cos': WarmUpCosLR,
}
