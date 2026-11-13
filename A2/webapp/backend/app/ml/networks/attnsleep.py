"""Adapter from this project's waveform tensor to the author's single-channel AttnSleep.

Ported from ``A2/model_lib.AttnSleep``. The author's architecture is vendored unchanged
in ``attnsleep_author.py`` together with its MIT license.
"""

import torch

from .attnsleep_author import AttnSleep as _AuthorAttnSleep


class AttnSleep(_AuthorAttnSleep):
    """Select one EEG channel and return to the 100 Hz rate the model was trained on.

    Input: ``[B, C, 30, 200]`` at 200 Hz in microvolts / 100. The serving pipeline low-pass
    filters at 35 Hz before upsampling, so keeping every second sample restores a 100 Hz
    epoch exactly as in training. This is not a general resampler.
    Output: ``[B, 5]`` logits.
    """

    def __init__(self, channel_index: int = 0):
        super().__init__()
        self.channel_index = channel_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().flatten(start_dim=2)
        x = x[:, self.channel_index:self.channel_index + 1]
        if x.shape[-1] != 6000:
            raise ValueError("Expected a 30-second epoch at 200 Hz.")
        return super().forward(x[..., ::2].contiguous())
