import numpy as np
import skimage
from skimage.draw import polygon
from skimage.draw import bezier_curve
from skimage.draw import circle_perimeter
from skimage.draw import disk
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt

import colorsys
import torch
ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0

def agt_color(aidx):
    base_color = matplotlib.colors.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][aidx % 10])
    
    # Increase brightness and make colors lighter
    h, l, s = colorsys.rgb_to_hls(*base_color)
    l = min(l + 0.3, 1.0)
    lighter_color = colorsys.hls_to_rgb(h, l, s)
    # return base_color    
    return lighter_color

def agt_color_new(aidx):
    base_color = matplotlib.colors.to_rgb(plt.rcParams['axes.prop_cycle'].by_key()['color'][aidx % 10])
    
    # Increase brightness and make colors lighter
    h, l, s = colorsys.rgb_to_hls(*base_color)
    l = min(l + 0.3, 1.0)
    lighter_color = colorsys.hls_to_rgb(h, l, s)
    # return base_color    
    return lighter_color

def get_base_colors(aidx_list, device='cpu'):
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_list = [matplotlib.colors.to_rgb(base_colors[idx % 10]) for idx in aidx_list]
    color_tensor = torch.tensor(color_list, dtype=torch.float32, device=device)  # shape: (N, 3)
    return color_tensor

def rgb_to_hls( rgb):
    maxc = torch.max(rgb, dim=-1)[0]
    minc = torch.min(rgb, dim=-1)[0]
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    h = torch.zeros_like(maxc)
    l = (minc+maxc)/2.0
    s = torch.zeros_like(maxc)
    mask = (minc == maxc)

    # if l <= 0.5:
    #     s = (maxc-minc) / (maxc+minc)
    mask2 = (l <= 0.5) & ~mask
    s[mask2] = ((maxc-minc) / (maxc+minc))[mask2]
    mask3 = ~mask2 & ~mask
    # else:
    #     s = (maxc-minc) / (2.0-maxc-minc)
    s[mask3] = ((maxc-minc) / (2.0-maxc-minc))[mask3]
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    mask4 = (r == maxc) & ~mask
    mask5 = (g == maxc) & ~mask
    mask6 = ~(mask4 | mask5) & ~mask
    h[mask4] = (bc-gc)[mask4]
    h[mask5] = (2.0+rc-bc)[mask5]
    h[mask6] = (4.0+gc-rc)[mask6]
    h = (h/6.0) % 1.0
    return h, l, s

def _v(m1, m2, hue):
    hue = hue % 1.0
    m = torch.zeros_like(m1)

    mask1 = (hue< ONE_SIXTH)
    m[mask1] = m1[mask1] + (m2[mask1]-m1[mask1])*hue[mask1]*6.0
    mask2 = (hue < 0.5) & ~mask1
    m[mask2] = m2[mask2]
    mask3 = (hue < TWO_THIRD) & ~mask1 & ~mask2
    m[mask3] = m1[mask3] + (m2[mask3]-m1[mask3])*(TWO_THIRD-hue[mask3])*6.
    mask4 = ~mask1 & ~mask2 & ~mask3
    m[mask4] = m1[mask4]
    return m


def hls_to_rgb( h, l, s):
    # if s == 0.0:
    #     return l, l, l
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    mask = (s == 0.0)
    r[mask] = l[mask]
    g[mask] = l[mask]
    b[mask] = l[mask]
    mask1 = (l<=0.5) & ~mask
    mask2 = ~mask1 & ~mask
    m2 = torch.zeros_like(l)
    m2[mask1] = l[mask1] * (1.0+s[mask1])
    m2[mask2] = l[mask2] + s[mask2]-(l[mask2]*s[mask2])
    m1 =torch.zeros_like(l)
    m1[~mask] = (2.0*l[~mask] - m2[~mask])
    r[~mask] = _v(m1, m2, h+ONE_THIRD)[~mask]
    g[~mask] = _v(m1, m2, h)[~mask]
    b[~mask] = _v(m1, m2, h-ONE_THIRD)[~mask]
    return r,g,b
    
    

def update_color_new(force,base_color, max_force=1500):
    h,l,s = rgb_to_hls(base_color)
    magnitude = torch.norm(force,dim=-1)
    factor = torch.clamp(magnitude / max_force, min=None, max = 1.0)
    l = torch.clamp(l * (1 - 0.8*factor),min=0)
    s = torch.clamp(s * (1 + factor), max=1)

    adjusted_r, adjusted_g, adjusted_b = hls_to_rgb(h, l, s)

    return adjusted_r, adjusted_g, adjusted_b

def agt_colors_batch(aidx_list, device='cpu', lighten=0.3):
    rgb = get_base_colors(aidx_list, device=device)  # (N, 3)
    hls = rgb_to_hls(rgb)  # (N, 3)
    hls[..., 1] = torch.clamp(hls[..., 1] + lighten, 0.0, 1.0)
    lighter_rgb = hls_to_rgb(hls)  # (N, 3)
    return lighter_rgb

def update_colors_batch(forces, base_colors, max_force=1500.0):
    """
    forces: (N, 3) tensor
    base_colors: (N, 3) RGB tensor, values in [0, 1]
    returns: (N, 3) adjusted RGB tensor
    """
    hls = rgb_to_hls(base_colors)  # (N, 3)
    h, l, s = hls.unbind(-1)

    magnitudes = torch.linalg.norm(forces, dim=-1)  # (N,)
    factors = torch.clamp(magnitudes / max_force, 0.0, 1.0)

    l_new = torch.clamp(l * (1 - 0.8 * factors), 0.0, 1.0)
    s_new = torch.clamp(s * (1 + factors), 0.0, 1.0)

    hls_adjusted = torch.stack([h, l_new, s_new], dim=-1)

    adjusted_rgb = hls_to_rgb(hls_adjusted)  # (N, 3)
    return adjusted_rgb

def draw_disk(img_size=80, max_r=10, iterations=3):
    shape = (img_size, img_size)
    img = np.zeros(shape, dtype=np.uint8)
    x, y = np.random.uniform(max_r, img_size - max_r, size=(2))
    radius = int(np.random.uniform(max_r))
    rr, cc = disk((x, y), radius, shape=shape)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_circle(img_size=80, max_r=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r, c = np.random.uniform(max_r, img_size - max_r, size=(2,)).astype(int)
    radius = int(np.random.uniform(max_r))
    rr, cc = circle_perimeter(r, c, radius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=1).astype(int)
    return img


def draw_curve(img_size=80, max_sides=10, iterations=3):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r0, c0, r1, c1, r2, c2 = np.random.uniform(0, img_size, size=(6,)).astype(int)
    w = np.random.random()
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, w)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    img = ndimage.binary_dilation(img, iterations=iterations).astype(int)
    return img


def draw_polygon(img_size=80, max_sides=10):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    num_coord = int(np.random.uniform(3, max_sides))
    r = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    c = np.random.uniform(0, img_size, size=(num_coord,)).astype(int)
    rr, cc = polygon(r, c)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img


def draw_ellipse(img_size=80, max_size=10):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    r, c, rradius, cradius = np.random.uniform(max_size, img_size - max_size), np.random.uniform(max_size, img_size - max_size),\
        np.random.uniform(1, max_size), np.random.uniform(1, max_size)
    rr, cc = skimage.draw.ellipse(r, c, rradius, cradius)
    np.clip(rr, 0, img_size - 1, out=rr)
    np.clip(cc, 0, img_size - 1, out=cc)
    img[rr, cc] = 1
    return img