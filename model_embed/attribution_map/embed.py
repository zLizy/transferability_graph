import numpy as np
from PIL import Image
import torch.nn.functional as F

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def main():
    elrp = np.zeros([args.imlist_size] + list(img.shape), float)
    saliency = np.zeros([args.imlist_size] + list(img.shape), float)
    gradXinput = np.zeros([args.imlist_size] + list(img.shape), float)

    saliency = Saliency(model)
    grads = saliency.attribute(input, target=pred_label_idx)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print(grads.shape)

_ = viz.visualize_image_attr_multiple(grads,
                                      np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True)

if __name__ == "__main__":
    main()