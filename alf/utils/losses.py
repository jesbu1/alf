# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various function/classes related to loss computation."""

import gin
import torch


@gin.configurable
def element_wise_huber_loss(x, y):
    """Elementwise Huber loss.

    Args:
        x (Tensor): label
        y (Tensor): prediction
    Returns:
        loss (Tensor)
    """
    loss = torch.nn.SmoothL1Loss(reduction="none")
    return loss(y, x)


@gin.configurable
def element_wise_squared_loss(x, y):
    """Elementwise squared loss.

    Args:
        x (Tensor): label
        y (Tensor): prediction
    Returns:
        loss (Tensor)
    """
    loss = torch.nn.MSELoss(reduction="none")
    return loss(y, x)
