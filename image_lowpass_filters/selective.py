'''
Author: Cassini Wei
Date: 2023/2/14
'''

import torch
from kornia import filters
from torch import Tensor
from torch.nn import functional
from typing import Tuple

#### Selective filters
# From: 
# Quiring E, Klein D, Arp D, et al. 
# Adversarial preprocessing: Understanding and preventing image-scaling attacks in machine learning[C]//
# Proceedings of the 29th USENIX Conference on Security Symposium. 2020: 1363-1380.


def _get_kernel_size(original_size: Tuple[int, int], input_size: Tuple[int, int] | None) -> Tuple[int, int]:
    """Compute the kernel size for selective filters.

    Args:
        original_size (Tuple[int, int]): (H, W).
        input_size (Tuple[int, int]): (H0, W0).

    Returns:
        Tuple[int, int]: [H1, W1].
    """    
    if input_size is None:
        input_size = original_size
    beta_h, beta_w = original_size[0] / input_size[0], original_size[1] / input_size[1]
    kernel_size = (int(beta_h * 2 + 1), int(beta_w * 2 + 1))
    return kernel_size


def selective_median(image: Tensor, input_size: Tuple[int, int] | None = None) -> Tensor:
    """Selective median filter.

    Args:
        image (Tensor): [B, C, H, W] or [C, H, W].
        input_size (Tuple[int, int], optional): Input size of the mmodel. Defaults to None.

    Returns:
        Tensor: [B, C, input_size[0], input_size[1]] or [C, input_size[0], input_size[1]].
    """    
    kernel_size = _get_kernel_size((image.shape[-2], image.shape[-1]), input_size)
    if image.dim() == 3:
        image_filtered = filters.median_blur(image.unsqueeze(0), kernel_size)[0]
    else:
        image_filtered = filters.median_blur(image, kernel_size)
    return image_filtered


def selective_random(image: Tensor, input_size: Tuple[int, int] | None = None) -> Tensor:
    """Selective random filter.

    Args:
        image (Tensor): [B, C, H, W] or [B, C, H, W]
        input_size (Tuple[int, int], optional): Input size of the mmodel. Defaults to None.

    Returns:
        Tensor: [B, C, input_size[0], input_size[1]] or [C, input_size[0], input_size[1]].
    """    
    kernel_size = _get_kernel_size((image.shape[-2], image.shape[-1]), input_size)
    orishape = image.shape
    if len(orishape) == 3:
        image = image.unsqueeze(0)
    n_batch, n_channel, h, w = image.shape
    n_pix_per_block, n_blocks = kernel_size[0] * kernel_size[1], h * w
    # Extract sliding local blocks and randomly select inside pixels
    local_blocks = functional.unfold(image, kernel_size, dilation=1, padding=1, stride=1) # [B, elements, blocks]
    local_blocks_c = local_blocks.reshape(
                        (n_batch, n_channel, kernel_size[0] * kernel_size[1], n_blocks)) # [B, C, elements_each_channel, blocks]
    random_indices = torch.randint(low=0, high=n_pix_per_block, size=(n_blocks,)).to(image.device)
    random_mask = functional.one_hot(random_indices).bool()
    image_rand = torch.masked_select(
                    local_blocks_c.permute(0, 1, 3, 2), 
                    random_mask
                ).reshape(n_batch, n_channel, h, w)
    if len(orishape) == 3:
        return image_rand[0]
    return image_rand

