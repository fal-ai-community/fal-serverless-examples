"""The four FlashPack + WAN 2.1 classes, exactly as FlashPack's flagship WAN
example defines them (github.com/fal-ai/flashpack, tests/test_wan_pipeline.py),
wrapped in a factory so this module's top level stays stdlib-only — the
deploying client and CI's static checks import this file without torch.

Shipped to the runner via `local_python_modules` (cloudpickle by-value). A
by-value module reconstructs WITHOUT a sys.modules entry, which breaks two
things this example needs:

  - transformers 5.x reads sys.modules[cls.__module__] when instantiating any
    PreTrainedModel subclass -> KeyError;
  - diffusers' pipeline loader resolves the component classes recorded in the
    pack's model_index.json via importlib.import_module("wan_flashpack")
    -> ModuleNotFoundError.

_publish_classes() repairs both at runtime by installing a real module entry.
"""
from __future__ import annotations

import sys
import types

_CLASSES: dict[str, type] = {}


def _publish_classes(module_name: str, classes: dict[str, type]) -> types.ModuleType:
    """Make `module_name` importable with `classes` as attributes.

    Reuses the existing sys.modules entry when the module was imported
    normally (local runs, tests); creates one when running from a cloudpickle
    by-value reconstruction (fal runner).
    """
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

    # Publish the module entry BEFORE defining the classes: transformers may
    # introspect the defining module during class creation, not just __init__.
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
        # rope buffers are recomputed at init; packing them would only bloat
        # the file (upstream flagship does the same).
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

    # diffusers' get_class_obj_and_candidates() strips a "FlashPack" prefix
    # from the class names recorded in model_index.json before resolving
    # them against this module (pipeline_loading_utils.py, diffusers 0.39):
    # "FlashPackUMT5EncoderModel" is looked up here as "UMT5EncoderModel".
    # Register every class under both names so either resolution path lands
    # on the mixin subclass.
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
