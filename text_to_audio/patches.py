"""Patches for Chatterbox TTS compatibility issues."""

import torch

_patches_applied = False


class DummyWatermarker:
    def apply(self, audio, sr):
        return audio
    
    def apply_watermark(self, signal,  sample_rate, **_):
        return signal
    
    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        return wm_signal


def apply_patches(device: str = "cpu") -> None:
    """Apply all necessary patches for Chatterbox TTS.

    Args:
        device: Target device for torch.load map_location patch.
    """
    global _patches_applied
    if _patches_applied:
        return

    # Patch 1: Fix perth watermarker issue (missing setuptools with uv)
    import perth
    perth.PerthImplicitWatermarker = DummyWatermarker

    # Patch 2: Fix torch.load map_location for non-CUDA devices
    map_location = torch.device(device)
    torch_load_original = torch.load

    def patched_torch_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = map_location
        return torch_load_original(*args, **kwargs)

    torch.load = patched_torch_load

    _patches_applied = True
