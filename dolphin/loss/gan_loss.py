import torch
import torch.nn as nn
import torch.nn.functional as F

from dolphin.utils import Registers
from .base import BaseWeightedLoss


@Registers.loss.register
class GANLoss(BaseWeightedLoss):
    """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports 
                               vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
              LSGAN needs no sigmoid. vanilla GANs will handle it with 
              BCEWithLogitsLoss.
        """

    def __init__(self, 
                 gan_mode,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 loss_weight=1.0):
        
        super(GANLoss, self).__init__(loss_weight=loss_weight)
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a 
                                    discriminator
            target_is_real (bool) - - if the ground truth label is 
                                      for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with 
            the size of the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def _forward(self, prediction, target_is_real, **kwargs):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a 
                                    discriminator
            target_is_real (bool) - - if the ground truth label is for real 
                                      images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        if self.gan_mode == 'lsgan':
            loss = F.mse_loss(prediction, target_tensor)
        elif self.gan_mode == 'vanilla':
            loss = F.binary_cross_entropy_with_logits(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise NotImplementedError(f'gan mode {self.gan_mode} not found.')
        return loss