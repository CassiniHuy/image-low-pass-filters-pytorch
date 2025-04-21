# Low-pass filters


Assume there is an image in spatial domain $f(u, v)\in\mathbb{R}^{m\times n}$, 
and its representation in shifted frequency domain $F(u, v)$, 
therefore the low-pass filtering is $H(u,v)*F(u, v)$, where

## Ideal

$$H(u, v)=\begin{cases}
    1, D(u, v) < D_0 \\
    0, D(u, v) > D_0
\end{cases}$$

where $D(u,v)$ is the distance to the matrix center for each pixel, and $D_0$ is the cutoff frequency.

## Butterworth

$$H(u, v)=\frac{1}{1+[D(u, v)/D_0]^{2n}}$$

## Gaussian

$$H(u, v)=e^{-D^2(u, v)/2{D_0}^2}$$

# Usage

Install from pypi:

```
pip install image-lowpass-filters
```

examples:

```python
import torch
from image_lowpass_filters import ideal_bandpass, butterworth, gaussian

cutoff = 20

img_tensor = torch.randn((1, 3, 224, 224))

img_lowpass = ideal_bandpass(img_tensor, cutoff)

img_lowpass = butterworth(img_tensor, cutoff, 10)

img_lowpass = gaussian(img_tensor, cutoff)
```

