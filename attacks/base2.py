# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta

import torch

from advertorch.utils import replicate_input


class Attack(object):
    """
    Abstract base class for all attack classes.

    :param predict: forward pass function.
    :param loss_fn: loss function that takes .
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.

    """

    __metaclass__ = ABCMeta

    def __init__(self, predict, loss_fn, clip_min, clip_max):
        """Create an Attack instance."""
        self.predict = predict
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max

    def perturb(self, x, **kwargs):
        """Virtual method for generating the adversarial examples.

        :param x: the model's input tensor.
        :param **kwargs: optional parameters used by child classes.
        :return: adversarial examples.
        """
        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)


class LabelMixin(object):
    # def _get_predicted_label(self, x):
    def _get_predicted_label(self, x, text, CTC):
        """
        Compute predicted labels given x. Used to prevent label leaking
        during adversarial training.

        :param x: the model's input tensor.
        :return: tensor containing predicted labels.
        """
        with torch.no_grad():
            # outputs = self.predict(x)
            if CTC:
                outputs = self.predict(x, text)
            else: 
                outputs = self.predict(x, text, is_train=False)
        # _, y = torch.max(outputs, dim=1)
        y = outputs.log_softmax(2).permute(1, 0, 2)
        return y

    # def _verify_and_process_inputs(self, x, y):
    def _verify_and_process_inputs(self, x, text, y, CTC):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                # y = self._get_predicted_label(x)
                y = self._get_predicted_label(x, text, CTC)

        x = replicate_input(x)
        y = replicate_input(y)
        return x, y
