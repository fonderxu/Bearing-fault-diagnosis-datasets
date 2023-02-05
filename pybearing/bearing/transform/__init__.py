import torch
import random
import numbers
from torch import Tensor
from typing import Sequence
import torchvision
from torch.nn import functional as F
from torch.nn.functional import grid_sample, conv2d, conv1d, interpolate, pad as torch_pad
import warnings
import torchaudio.transforms as T

warnings.filterwarnings("ignore", category=DeprecationWarning)

__all__ = [
    'Compose',
    "Normalization",

    "HShift",
    "VShift",
    "HFlip",
    "VFlip",

    "RandomTimeMasking",
    "TimeMasking",
    "GaussianNoise",
    "RandomScaling",

    "Rotate",
]


class Compose(torchvision.transforms.Compose):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)


class Normalization(torch.nn.Module):
    def __int__(self) -> None:
        super(Normalization, self).__int__()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        _min, _ = torch.min(x, -1)
        _max, _ = torch.max(x, -1)
        _min, _max = _min.unsqueeze(-1), _max.unsqueeze(-1)
        return -1. + 2 * (x - _min) / (_max - _min)


class RandomScaling(torch.nn.Module):
    def __init__(self, p=0.5) -> None:
        super(RandomScaling, self).__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = x.clone()
        '''
        scaling the amplify of signal
        '''
        b = x.shape[0]
        random_seed = torch.rand(b, device=x.device) < self.p
        scale_ratio = torch.rand(b, device=x.device) + 0.5
        scaled_x = []
        for index in range(b):
            scaled_x.append(x[index] * scale_ratio[index] if random_seed[index] else x[index])
        return torch.stack(scaled_x)


def make_mask(x: Tensor, start: int, end: int) -> Tensor:
    mask = torch.ones_like(x, device=x.device)
    if start <= end:
        mask[..., start: end] = 0.
    else:
        mask[..., :end] = mask[..., start:] = 0.
    return mask


class RandomTimeMasking(torch.nn.Module):
    """
    Single sample augment
    Random Crop consequent vector of length of size from original vector,  Cropped areas padded with zero
    Args:
        snr (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, min_time=100, max_time=500, p=0.25) -> None:
        super(RandomTimeMasking, self).__init__()
        self.min_time = min_time
        self.max_time = max_time
        self.p = p

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = x.clone()
        """
        Args:
            x:
        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        b, w = x.shape[0], x.shape[-1]
        random_seed = torch.rand(b) < self.p
        masked_x = []
        for index in range(b):
            if not random_seed[index]:
                masked_x.append(x[index])
            else:
                time_mask = random.randint(self.min_time, self.max_time)
                time_start = random.randint(0, w)
                mask = make_mask(x[index], time_start, (time_start + time_mask) % w)
                masked_x.append(mask * x[index])
        return torch.stack(masked_x)


class GaussianNoise(torch.nn.Module):
    """
    Single sample augment
    Args:
        snr (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, snr=20., p=0.5) -> None:
        super().__init__()
        self.snr = snr
        self.p = p

    @torch.no_grad()
    def forward(self, x):
        '''
        Returns: noise + x
        '''
        b, w = x.shape[0], x.shape[-1]
        random_seed = (torch.rand(list(x.shape[:-1]) + [1], device=x.device) < self.p).type(x.dtype)

        Ps = torch.sum(x ** 2, dim=0) / w
        Pn = Ps / (10 ** (self.snr / 10))
        noise = torch.randn_like(x, device=x.device) * torch.sqrt(Pn).expand_as(x)
        return x + random_seed * noise


class HShift(torch.nn.Module):
    """
    Single sample augment
    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, x) -> Tensor:
        b, w = x.shape[0], x.shape[-1]
        rolls = []
        random_seed = torch.rand(b) < self.p
        for index in range(b):
            rolls.append(torch.roll(x[index], random.randint(0, w), dims=-1) if random_seed[index] else x[index])
        return torch.stack(rolls)


class VShift(torch.nn.Module):
    """
    Single sample augment
    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, x):
        _mean = torch.mean(x, dim=-1).unsqueeze(dim=-1)
        _max, _ = torch.max(torch.abs(x - _mean), dim=-1)

        random_seed = (torch.rand_like(_max, device=_max.device) < self.p).float()
        value_float = (torch.randn_like(_max, device=_max.device) * (_max / 2) * random_seed).unsqueeze(
            dim=-1).expand_as(
            x)
        return x + value_float


class VFlip(torch.nn.Module):
    """
    Single sample augment
    Args:
        snr (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x:
        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        b = x.shape[0]
        random_seed = torch.rand(b) < self.p
        flip = []
        for index in range(b):
            flip.append(x[index].neg() if random_seed[index] else x[index])
        return torch.stack(flip)


class HFlip(torch.nn.Module):
    """
    Single sample augment
    Args:
        snr (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            img (PIL Image or Tensor): Image to be solarized.

        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        b = x.shape[0]
        random_seed = torch.rand(b) < self.p
        flip = []
        for index in range(b):
            flip.append(x[index].flip(-1) if random_seed[index] else x[index])
        return torch.stack(flip)


@DeprecationWarning
def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


@DeprecationWarning
class Rotate(torch.nn.Module):
    """
    Single sample augment
    Args:
        snr (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being color inverted. Default value is 0.5
        random rotate between +15 and -15 angle
    """

    def __init__(self, p=0.5, max_angle=5, rotate_loc=None) -> None:
        super().__init__()
        self.p = p
        self.x = None
        self.max_angle = max_angle
        self.rotate_loc = rotate_loc

    @torch.no_grad()
    def forward(self, y):
        """
        Args:
            img (PIL Image or Tensor): Image to be solarized.

        Returns:
            PIL Image or Tensor: Randomly solarized image.
        """
        if random.uniform(0, 1) < self.p:
            # if self.rotate_loc is None:
            # self.rotate_loc = torch.randint(0, y.shape[-1], (1,)).item()
            rotate_loc = random.randint(0, y.shape[-1])
            if self.x is None:
                self.x = torch.arange(0 - rotate_loc, y.shape[-1] - rotate_loc, device=y.device).expand_as(y)
            random_angle = (torch.pi * 2 / 360 * self.max_angle * torch.rand(1, device=y.device)).expand_as(y)
            return self.x * torch.sin(random_angle) + y * torch.cos(random_angle)
        else:
            return y


@DeprecationWarning
class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, p, sigma=(0.1, 2.0)) -> None:
        super().__init__()
        self.p = p
        self.kernel_size = kernel_size
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0.0 < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    @torch.no_grad()
    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        if random.uniform(0, 1) <= self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            kernel = _get_gaussian_kernel1d(self.kernel_size, sigma)
            kernel = kernel.expand(1, 1, 1, self.kernel_size).to(img.device)
            padding = [self.kernel_size // 2, self.kernel_size // 2, 0, 0]
            img = torch_pad(img, padding, mode="reflect")
            img = conv2d(img, kernel, groups=img.shape[-3])
            return img
        else:
            return img

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma})"
        return s


# @DeprecationWarning
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class TimeMasking(torch.nn.Module):
    def __init__(self, block_size, ratio, flip_p, shift_p, norm) -> None:
        super(TimeMasking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.transform = Compose(
            [
                VFlip(flip_p),
                HFlip(flip_p),
                HShift(shift_p)
            ]
        )
        if norm: self.transform.transforms.append(Normalization())

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = self.transform(x.clone())
        B, W = x.shape[0], x.shape[-1]
        mshape = B, 1, round(W / self.block_size)
        time_mask = torch.rand(mshape, device=x.device)
        time_mask = (time_mask > self.ratio).float()
        time_mask = resize(time_mask, size=W)
        return x * time_mask.view_as(x)


@DeprecationWarning
class Resample(torch.nn.Module):
    def __init__(
            self,
            orig_freq: int = 12000,
            resampling_method: str = "sinc_interpolation",
            lowpass_filter_width: int = 6,
            rolloff: float = 0.99,
    ) -> None:
        super().__init__()
        self.orig_freq = orig_freq
        self.resample_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff

    def forward(self, x: Tensor) -> Tensor:
        b = x.shape[0]
        resampled_x = []
        # 512 < resample_ratio / self.orig_freq * 1024 < 1536 ->
        random_seed = torch.rand(b, device=x.device) * 0.4 + 0.8
        for index in range(b):
            resample_x_index = T.Resample(self.orig_freq, (self.orig_freq * random_seed[index]).int())(x[index])
            resampled_x.append(resample_x_index if random_seed[index] >= 1. else torch.cat(
                (resample_x_index, resample_x_index.flip(-1)), dim=-1))
        return torch.stack(resampled_x)


def random_select(*perturbed_samples_set):
    num_perturbed = len(perturbed_samples_set)
    for i in range(1, num_perturbed):
        assert perturbed_samples_set[i - 1].shape == perturbed_samples_set[i].shape
    batch_size = perturbed_samples_set[0].shape[0]
    random_selected = []
    random_seeds = torch.randint(0, num_perturbed, (batch_size,))
    for i in range(batch_size):
        random_selected.append(perturbed_samples_set[random_seeds[i]][i].unsqueeze(0))
    return torch.cat(random_selected, dim=0)
