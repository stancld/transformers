# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from tests.test_modeling_common import floats_tensor
import unittest

import numpy as np

from transformers import Speech2TextConfig, is_tf_available
from transformers.file_utils import cached_property
from transformers.models.speech_to_text.modeling_tf_speech_to_text import TFSpeech2TextModel
from transformers.testing_utils import require_tf, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor, floats_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import TFSpeech2TextPreTrainedModel


def prepare_speech_to_text_inputs_dict(
    config,
    input_features,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = tf.cast(tf.math.not_equal(input_features, 0), tf.int8)
    if decoder_attention_mask is None:
        decoder_attention_mask = tf.cast(tf.math.not_equal(decoder_input_ids, config.pad_token_id), tf.int8)
    if head_mask is None:
        head_mask = tf.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = tf.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


@require_tf
class TFSpeech2TextModelTester:
    config_cls = Speech2TextConfig
    config_updates = {}
    hidden_act = "relu"

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        num_conv_layers=2,
        conv_kernel_sizes=(5, 5),
        conv_channels=32,
        input_feat_per_channel=24,
        input_channels=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=20,
        max_target_positions=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_channels = conv_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs_for_common(self):
        input_features = floats_tensor(
            [self.batch_size, self.seq_length, self.input_feat_per_channel], self.vocab_size
        )
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.config_cls(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            num_conv_layers=self.num_conv_layers,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_channels=self.conv_channels,
            input_feat_per_channel=self.input_feat_per_channel,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

        inputs_dict = prepare_speech_to_text_inputs_dict(
            config, input_features=input_features, decoder_input_ids=decoder_input_ids,
        )
        return config, inputs_dict


@require_tf
class TFSpeech2TextModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFSpeech2TextModel,) if is_tf_available() else ()
    all_generative_model_classes = ()
    is_encoder_decoder = True
    test_pruning = False
    test_onnx = True
    onnx_min_opset = 10

    def setUp(self):
        self.model_tester = TFSpeech2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Speech2TextConfig)
