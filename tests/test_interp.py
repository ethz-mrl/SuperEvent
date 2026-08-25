import torch

from models.util import interpolate_desc_grid

def pixel_to_normalized(px, shape):
    px = px.float()
    px[..., 0] = (px[..., 0] + 0.5) / shape[-2] * 2 - 1
    px[..., 1] = (px[..., 1] + 0.5) / shape[-1] * 2 - 1
    return px

shape = [1, 256, 480, 640]
grid_size = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
kpts = torch.rand([500, 2]).to(device)
kpts[:, 0] *= shape[-2]
kpts[:, 1] *= shape[-1]
kpts = kpts.to(torch.int).to(device)
desc_grid = torch.rand(shape[:2] + [shape[-2] // 8, shape[-1] // 8]).to(device)

# Ground truth
desc_gt = torch.nn.functional.interpolate(desc_grid, scale_factor=8, mode="bilinear")
desc_gt = desc_gt[:, :, kpts[:, 0], kpts[:, 1]]
desc_gt = torch.nn.functional.normalize(desc_gt, dim=1)

# More efficient
desc_new = interpolate_desc_grid(desc_grid, kpts, shape)

# Eval
assert (desc_gt - desc_new).abs().max() < 1e-5
print("Test passed.")
