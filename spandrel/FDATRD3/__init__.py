from spandrel.util import KeyCondition

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.fdat_rd3 import FDATRD3, RD3_META_KEY, describe_resize, fdat_rd3_config_from_state_dict


class FDATRD3Arch(Architecture[FDATRD3]):
    """FDATRD3: FDAT-RD2 / FDAT-RD 2x network with an inference-optimised graph.
    Detected by the 'MetaFDATRD3' marker. FDATRD3 files carry no 'upsampler.MetaUpsample', which
    the FDATRD and FDATRD2 detectors require, so the three never claim each other's files."""

    def __init__(self):
        super().__init__(
            id="FDATRD3",
            detect=KeyCondition.has_all(
                RD3_META_KEY,
                "conv_first.1.weight",
                "groups.0.blocks.0.n1.weight",
                "groups.0.blocks.0.ffn.fc1.weight",
                "upsampler.0.weight",
                "upsampler.2.weight",
                "upsampler.3.weight",
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[FDATRD3]:
        cfg = fdat_rd3_config_from_state_dict(state_dict)
        model = FDATRD3(**cfg)
        sizes = {96: "small", 128: "medium", 192: "large"}
        tags = [f"{cfg['embed_dim']}dim", f"stem-{cfg['stem']}"]
        if cfg["stem"] == "resize":
            tags.append(f"pre-{describe_resize(model.pre.MetaResize)}")
        if cfg["embed_dim"] in sizes:
            tags.append(sizes[cfg["embed_dim"]])
        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="SR",
            tags=tags,
            supports_half=True,
            supports_bfloat16=True,
            scale=2,
            input_channels=cfg["num_in_ch"],
            output_channels=cfg["num_out_ch"],
        )


__all__ = ["FDATRD3Arch", "FDATRD3"]
