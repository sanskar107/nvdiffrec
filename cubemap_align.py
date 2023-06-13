import torch
import numpy as np
import nvdiffrast.torch as dr
from render import util


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    R = torch.zeros((3, 3), dtype=torch.float32, device='cuda')
    R[0, 1] = -1
    R[1, 2] = 1
    R[2, 0] = -1
    # R[0, 2] = -1
    # R[1, 0] = -1
    # R[2, 1] = 1
    print(R)

    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))
        v = (R @ v[..., None])[..., 0]

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def cubemap_to_latlong(cubemap, res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return dr.texture(cubemap[None, ...], reflvec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
lgt[5] = 1.0
plus_x = cubemap_to_latlong(lgt, (256, 512)) * 255
cubemap = latlong_to_cubemap(plus_x, (256, 256))
print(cubemap.shape)
for i in range(6):
    print(cubemap[i].sum() / cubemap.sum())

# lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
# lgt[0] = 1.0
# env = cubemap_to_latlong(lgt, (256, 512)) * 255
# util.save_image_raw('env_img/plus_x.png', env.detach().cpu().numpy())

# lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
# lgt[1] = 1.0
# env = cubemap_to_latlong(lgt, (256, 512)) * 255
# util.save_image_raw('env_img/minus_x.png', env.detach().cpu().numpy())

# lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
# lgt[2] = 1.0
# env = cubemap_to_latlong(lgt, (256, 512)) * 255
# util.save_image_raw('env_img/plus_y.png', env.detach().cpu().numpy())

# lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
# lgt[3] = 1.0
# env = cubemap_to_latlong(lgt, (256, 512)) * 255
# util.save_image_raw('env_img/minus_y.png', env.detach().cpu().numpy())

# lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
# lgt[4] = 1.0
# env = cubemap_to_latlong(lgt, (256, 512)) * 255
# util.save_image_raw('env_img/plus_z.png', env.detach().cpu().numpy())

# lgt = torch.zeros((6, 256, 256, 3), dtype=torch.float32, device='cuda')
# lgt[5] = 1.0
# env = cubemap_to_latlong(lgt, (256, 512)) * 255
# util.save_image_raw('env_img/minus_z.png', env.detach().cpu().numpy())
