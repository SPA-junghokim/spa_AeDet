# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import math
import pdb

def gaussian_2d(shape, sigma1=1, sigma2=1):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma1 * sigma2))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, gauss_radi, k=1, pc_range=None, feature_map_size=None, radius=None,azi_gaus_thr=0):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = int(2 * gauss_radi + 1)  # ori
    temp_rad = max(azi_gaus_thr, int(torch.atan2( 2*torch.tensor(gauss_radi), radius)+math.pi * (  2*math.pi/feature_map_size[1] ) * ( 2 *gauss_radi)))
    #temp_rad = max(azi_gaus_thr, int(torch.atan2( 2*torch.tensor(gauss_radi), radius)+math.pi * (  2*math.pi/feature_map_size[1] ) * ( 2 *gauss_radi)))
    diameter2 = int(2 * temp_rad + 1)
    
    gaussian = gaussian_2d((diameter, diameter2), sigma1=diameter / 6, sigma2=diameter2 / 6)

    r, a = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2] # 256, 64

    left, right = min(r, gauss_radi), min(width - r, gauss_radi + 1)
    top, bottom = min(a, temp_rad), min(height - a, temp_rad + 1)
    
    masked_heatmap = heatmap[a - top:a + bottom,  r - left:r + right]
    masked_gaussian = torch.from_numpy(
        gaussian[#gauss_radi - top:gauss_radi + bottom,
                 gauss_radi - left:gauss_radi + right,
                 temp_rad - top:temp_rad + bottom,
                 ]).to(heatmap.device,torch.float32).T
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        
    return heatmap


def gaussian_radius_polar(det_size, min_overlap=0.5, y=None):
    """Get radius of gaussian.

    Args:
        det_size (tuple[torch.Tensor]): Size of the detection result.
        min_overlap (float, optional): Gaussian_overlap. Defaults to 0.5.

    Returns:
        torch.Tensor: Computed radius.
    """
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    # chgd
    res = min(r1, r2, r3)
    #res = res * (32 / y)
    return res


def get_ellip_gaussian_2D(heatmap, center, radius_x, radius_y, k=1):
    """Generate 2D ellipse gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius_x (int): X-axis radius of gaussian kernel.
        radius_y (int): Y-axis radius of gaussian kernel.
        k (int, optional): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian_kernel = ellip_gaussian2D((radius_x, radius_y),
                                       sigma_x=diameter_x / 6,
                                       sigma_y=diameter_y / 6,
                                       dtype=heatmap.dtype,
                                       device=heatmap.device)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def ellip_gaussian2D(radius,
                     sigma_x,
                     sigma_y,
                     dtype=torch.float32,
                     device='cpu'):
    """Generate 2D ellipse gaussian kernel.

    Args:
        radius (tuple(int)): Ellipse radius (radius_x, radius_y) of gaussian
            kernel.
        sigma_x (int): X-axis sigma of gaussian function.
        sigma_y (int): Y-axis sigma of gaussian function.
        dtype (torch.dtype, optional): Dtype of gaussian tensor.
            Default: torch.float32.
        device (str, optional): Device of gaussian tensor.
            Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius_y + 1) * (2 * radius_x + 1)`` shape.
    """
    x = torch.arange(
        -radius[0], radius[0] + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius[1], radius[1] + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x) / (2 * sigma_x * sigma_x) - (y * y) /
         (2 * sigma_y * sigma_y)).exp()
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0

    return h
