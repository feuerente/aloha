from torchvision import transforms
from typing import Dict, Tuple, Optional, List, Union, Any
import torch
import numpy as np

   
def compose_image_transform(image_observation,image_key: str, image_size: Optional[List[int]], crop_size: Optional[List[int]], random_crop: bool, normalize: bool) -> Tuple[transforms.Compose, Tuple[int, int, int]]:
    image_transform_list = []
    if normalize:
        # Divide by 255 to scale to [0, 1]
        image_transform_list.append(transforms.Lambda(lambda x: x / 255.0))
    # Get tensor image
    tensor_image = torch.tensor(image_observation[image_key])
    assert isinstance(tensor_image, torch.Tensor), f"Image must be a tensor, but is {type(tensor_image)}."
    # Determine original image shape
    org_image_shape = tensor_image.shape
    assert len(org_image_shape) == 3, f"Image must have 3 dimensions, but has {len(org_image_shape)}."
    assert org_image_shape[0] == 3, f"Image must have 3 channels, but has {org_image_shape[0]} in shape {org_image_shape}."
    # Resizing images
    if image_size is None:
        image_shape = org_image_shape
    else:
        assert isinstance(image_size, list) and len(image_size) == 2
        image_shape = tuple([3] + image_size)
        if image_size != list(org_image_shape[1:]):
            image_transform_list.append(transforms.Resize(image_size, antialias=True))
    # (Random) cropping images
    if crop_size is not None:
        if not crop_size[0] <= image_shape[1] and crop_size[1] <= image_shape[2]:
            raise ValueError(f"Crop size {crop_size} is larger than image size {image_shape[1:]}.")
        image_shape = tuple([3] + crop_size)
        if random_crop:
            image_transform_list.append(transforms.RandomCrop(crop_size))
        else:
            image_transform_list.append(transforms.CenterCrop(crop_size))
    return transforms.Compose(image_transform_list), image_shape

def adjust_images(image_observation,keys):
    for key in keys:
       if image_observation[key].shape[0] == 3:
           continue
       image_observation[key] =  np.moveaxis(image_observation[key],-1,0)
    