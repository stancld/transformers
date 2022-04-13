# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_longt5": ["LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongT5Config", "LongT5OnnxConfig"],
}

if is_torch_available():
    _import_structure["modeling_longt5"] = [
        "LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LongT5EncoderModel",
        "LongT5ForConditionalGeneration",
        "LongT5Model",
        "LongT5PreTrainedModel",
        "load_tf_weights_in_longt5",
    ]

if is_tf_available():
    _import_structure["modeling_tf_longt5"] = [
        "TF_LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLongT5EncoderModel",
        "TFLongT5ForConditionalGeneration",
        "TFLongT5Model",
        "TFLongT5PreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_longt5"] = [
        "FlaxLongT5ForConditionalGeneration",
        "FlaxLongT5Model",
        "FlaxLongT5PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_longt5 import LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, LongT5Config, LongT5OnnxConfig

    if is_torch_available():
        from .modeling_longt5 import (
            LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            LongT5EncoderModel,
            LongT5ForConditionalGeneration,
            LongT5Model,
            LongT5PreTrainedModel,
            load_tf_weights_in_longt5,
        )

    if is_tf_available():
        from .modeling_tf_longt5 import (
            TF_LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLongT5EncoderModel,
            TFLongT5ForConditionalGeneration,
            TFLongT5Model,
            TFLongT5PreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_longt5 import (
            FlaxLongT5ForConditionalGeneration,
            FlaxLongT5Model,
            FlaxLongT5PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
