'''
Author: Cassini Wei
Date: 2023/2/16
'''

import torch
from torch import Tensor
from typing import Tuple, Union


### Ideal low-pass

def _get_center_distance(size: Tuple[int, int], device: Union[str, torch.device] = 'cpu') -> Tensor:
    """Compute the distance of each matrix element to the center.

    Args:
        size (Tuple[int, int]): [m, n].
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [m, n].
    """    
    m, n = size
    i_ind = torch.tile(
                torch.tensor([[[i]] for i in range(m)], device=device),
                dims=[1, n, 1]).float()  # [m, n, 1]
    j_ind = torch.tile(
                torch.tensor([[[i] for i in range(n)]], device=device),
                dims=[m, 1, 1]).float()  # [m, n, 1]
    ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
    ij_ind = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
    center_ij = torch.tensor(((m - 1) / 2, (n - 1) / 2), device=device).reshape(1, 2)
    center_ij = torch.tile(center_ij, dims=[m * n, 1, 1])
    dist = torch.cdist(ij_ind, center_ij, p=2).reshape([m, n])
    return dist


def _get_ideal_weights(size: Tuple[int, int], D0: Union[int, float], lowpass: bool = True, device: Union[str, torch.device] = 'cpu') -> Tensor:
    """Get H(u, v) of ideal bandpass filter.

    Args:
        size (Tuple[int, int]): [H, W].
        D0 (Union[int, float]): The cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size, device)
    center_distance[center_distance > D0] = -1
    center_distance[center_distance != -1] = 1
    if lowpass is True:
        center_distance[center_distance == -1] = 0
    else:
        center_distance[center_distance == 1] = 0
        center_distance[center_distance == -1] = 1
    return center_distance


def _to_freq(image: Tensor) -> Tensor:
    """Convert from spatial domain to frequency domain.

    Args:
        image (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W]
    """    
    img_fft = torch.fft.fft2(image)
    img_fft_shift = torch.fft.fftshift(img_fft)
    return img_fft_shift


def _to_space(image_fft: Tensor) -> Tensor:
    """Convert from frequency domain to spatial domain.

    Args:
        image_fft (Tensor): [B, C, H, W].

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_ifft_shift = torch.fft.ifftshift(image_fft)
    img_ifft = torch.fft.ifft2(img_ifft_shift)
    img = img_ifft.real.clamp(0, 1)
    return img


def ideal_bandpass(image: Tensor, D0: Union[int, float], lowpass: bool = True) -> Tensor:
    """Low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (Union[int, float]): Cutoff frequency.
        lowpass (bool): True for low-pass filter, otherwise for high-pass filter. Defaults to True.

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_fft = _to_freq(image)
    weights = _get_ideal_weights((img_fft.shape[-2], img_fft.shape[-1]), D0=D0, lowpass=lowpass, device=image.device)
    img_fft = img_fft * weights
    img = _to_space(img_fft)
    return img

#### Butterworth

def _get_butterworth_weights(size: Tuple[int, int], D0: Union[int, float], n: int, device: Union[str, torch.device] = 'cpu') -> Tensor:
    """Get H(u, v) of Butterworth filter.

    Args:
        size (Tuple[int, int]): [H, W].
        D0 (Union[int, float]): The cutoff frequency.
        n (int): Order of Butterworth filters.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size=size, device=device)
    weights = 1 / (1 + torch.pow(center_distance / D0, 2 * n))
    return weights


def butterworth(image: Tensor, D0: Union[int, float], n: int) -> Tensor:
    """Butterworth low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (Union[int, float]): Cutoff frequency.
        n (int): Order of the Butterworth low-pass filter.

    Returns:
        Tensor: [B, C, H, W].
    """    
    img_fft = _to_freq(image)
    weights = _get_butterworth_weights((image.shape[-2], image.shape[-1]), D0, n, device=image.device)
    img_fft = weights * img_fft
    img = _to_space(img_fft)
    return img


#### Gaussian


def _get_gaussian_weights(size: Tuple[int, int], D0: Union[int, float], device: Union[str, torch.device] = 'cpu') -> Tensor:
    """Get H(u, v) of Gaussian filter.

    Args:
        size (Tuple[int, int]): [H, W].
        D0 (Union[int, float]): The cutoff frequency.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size=size, device=device)
    weights = torch.exp(- (torch.square(center_distance) / (2 * D0 ** 2)))
    return weights


def gaussian(image: Tensor, D0: Union[int, float]) -> Tensor:
    """Gaussian low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (Union[int, float]): Cutoff frequency.

    Returns:
        Tensor: [B, C, H, W].
    """    
    weights = _get_gaussian_weights((image.shape[-2], image.shape[-1]), D0=D0, device=image.device)
    image_fft = _to_freq(image)
    image_fft = image_fft * weights
    image = _to_space(image_fft)
    return image
