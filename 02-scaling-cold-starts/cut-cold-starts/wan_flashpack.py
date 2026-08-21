"""FlashPack + WAN 2.1 integration classes, built lazily so the top level stays stdlib-only."""
from __future__ import annotations

import sys
import types

_CLASSES: dict[str, type] = {}


def _publish_classes(module_name: str, classes: dict[str, type]) -> types.ModuleType:
    """Make `module_name` importable: cloudpickle by-value reconstruction leaves no sys.modules entry."""
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
    for name, cls in classes.items():
        setattr(module, name, cls)
    return module


def get_classes() -> dict[str, type]:
    """Build (once per process) and return the four FlashPack WAN classes."""
    if _CLASSES:
        return _CLASSES

    # Publish BEFORE defining the classes: transformers may introspect the
    # defining module during class creation.
    _publish_classes(__name__, {})

    from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
    from diffusers.pipelines import WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler
    from flashpack.integrations.diffusers import (
        FlashPackDiffusersModelMixin,
        FlashPackDiffusionPipeline,
    )
    from flashpack.integrations.transformers import FlashPackTransformersModelMixin
    from transformers import AutoTokenizer, UMT5EncoderModel

    class FlashPackWanTransformer3DModel(
        WanTransformer3DModel, FlashPackDiffusersModelMixin
    ):
        # rope buffers are recomputed at init; packing them only bloats the file.
        flashpack_ignore_prefixes = ["rope"]

    class FlashPackAutoencoderKLWan(AutoencoderKLWan, FlashPackDiffusersModelMixin):
        pass

    class FlashPackUMT5EncoderModel(UMT5EncoderModel, FlashPackTransformersModelMixin):
        # tied to shared.weight; transformers re-ties after load.
        flashpack_ignore_names = ["encoder.embed_tokens.weight"]

    class FlashPackWanPipeline(WanPipeline, FlashPackDiffusionPipeline):
        def __init__(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: FlashPackUMT5EncoderModel,
            vae: FlashPackAutoencoderKLWan,
            scheduler: UniPCMultistepScheduler,
            transformer: FlashPackWanTransformer3DModel | None = None,
            transformer_2: FlashPackWanTransformer3DModel | None = None,
            boundary_ratio: float | None = None,
            expand_timesteps: bool = False,
        ):
            super().__init__(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                transformer=transformer,
                transformer_2=transformer_2,
                scheduler=scheduler,
                boundary_ratio=boundary_ratio,
                expand_timesteps=expand_timesteps,
            )

    # diffusers 0.39 strips the "FlashPack" prefix when resolving
    # model_index.json class names - register every class under both names.
    mod = sys.modules[__name__]
    for cls_obj in (
        FlashPackWanTransformer3DModel,
        FlashPackAutoencoderKLWan,
        FlashPackUMT5EncoderModel,
        FlashPackWanPipeline,
    ):
        setattr(mod, cls_obj.__name__, cls_obj)
        setattr(mod, cls_obj.__name__.removeprefix("FlashPack"), cls_obj)
        _CLASSES[cls_obj.__name__] = cls_obj

    return _CLASSES
