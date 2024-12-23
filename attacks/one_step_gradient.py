# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import batch_multiply

from .base2 import Attack
from .base2 import LabelMixin


class GradientSignAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    # def perturb(self, x, y=None):
    def perturb(self, x, text=None, y=None, length=None, batch_size = 192, text_loss=None, length_loss=None, CTC=True):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, text, y, CTC)
        xadv = x.requires_grad_()
        
        if text is not None:
            if CTC:
                outputs = self.predict(xadv, text)
                preds_size = torch.IntTensor([outputs.size(1)] * batch_size)
                outputs = outputs.log_softmax(2).permute(1, 0, 2)
                if text_loss is not None:
                    loss = self.loss_fn(outputs, text_loss, preds_size, length_loss) # change
                else:
                    loss = self.loss_fn(outputs, text, preds_size, length)
            else:
                outputs = self.predict(xadv, text, is_train=False)
                if text_loss is not None:
                    outputs = outputs[:, :text_loss.shape[1] - 1, :]
                    target = text_loss[:, 1:]  # without [GO] Symbol
                    loss = self.loss_fn(outputs.contiguous().view(-1, outputs.shape[-1]), target.contiguous().view(-1))
                else:
                    target = text[:, 1:]  # without [GO] Symbol
                    loss = self.loss_fn(outputs.view(-1, outputs.shape[-1]), target.contiguous().view(-1))
        else:
            outputs = self.predict(xadv)
            loss = self.loss_fn(outputs, y)
            
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_sign = xadv.grad.detach().sign()

        xadv = xadv + batch_multiply(self.eps, grad_sign)
        # print(grad_sign, xadv)

        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()


FGSM = GradientSignAttack


class GradientAttack(Attack, LabelMixin):
    """
    Perturbs the input with gradient (not gradient sign) of the loss wrt the
    input.

    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.3,
                 clip_min=0., clip_max=1., targeted=False):
        """
        Create an instance of the GradientAttack.
        """
        super(GradientAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.

        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad = normalize_by_pnorm(xadv.grad)
        xadv = xadv + batch_multiply(self.eps, grad)
        xadv = clamp(xadv, self.clip_min, self.clip_max)

        return xadv.detach()


FGM = GradientAttack
