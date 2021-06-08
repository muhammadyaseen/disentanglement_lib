# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Hyperparameter sweeps and configs for the study "unsupervised_study_v1".

Challenging Common Assumptions in the Unsupervised Learning of Disentangled
Representations. Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch,
Sylvain Gelly, Bernhard Schoelkopf, Olivier Bachem. arXiv preprint, 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
import disentanglement_lib.utils.hyperparams as h
from six.moves import range


def get_datasets():
    """Returns all the data sets with corresponding correlation indices."""

    # Shapes3D A
    correlation_indices = h.fixed("correlation_details.corr_indices", [3, 5])
    dataset_name = h.fixed("dataset.name", "shapes3d")
    config_shapes3d_size_azimuth = h.zipit([correlation_indices, dataset_name])

    all_datasets = h.chainit([
        config_shapes3d_size_azimuth
    ])
    return all_datasets


def get_num_latent(sweep):
    return h.sweep("encoder.num_latent", h.discrete(sweep))


def get_line_widths():
    """Returns random seeds."""
    return h.sweep("correlation_hyperparameter.line_width", h.discrete([0.1, 0.2, 0.4, 0.7, 10.0]))


def get_seeds(num):
    """Returns random seeds."""
    return h.sweep("model.random_seed", h.categorical(list(range(num))))


def get_default_models():
    """Our default set of models (6 model * 6 hyperparameters=36 models)."""
    # BetaVAE config.
    model_name = h.fixed("model.name", "group_vae_argmax")
    model_fn = h.fixed("model.model", "@group_vae_argmax()")
    betas = h.sweep("group_vae_argmax.beta", h.discrete([1., 2., 4., 6., 8., 16.]))
    config_beta_vae = h.zipit([model_name, betas, model_fn])

    all_models = h.chainit([
        config_beta_vae
    ])
    return all_models


def get_correlation_types():
    """Returns all types of correlation"""
    return h.sweep(
        "correlation_details.corr_type",
        h.categorical([
            "line"
        ]))


def get_config():
    """Returns the hyperparameter configs for different experiments."""
    arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
    arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
    corr_act = h.fixed("correlation.active_correlation", True)
    architecture = h.zipit([arch_enc, arch_dec, corr_act])
    return h.product([
        get_datasets(),
        get_line_widths(),
        get_correlation_types(),
        architecture,
        get_default_models(),
        get_seeds(10)
    ])


class CorrelatedFactorsStudyWSID2(study.Study):
    """Defines the study for the paper."""

    def get_model_config(self, model_num=0):
        """Returns model bindings and config file."""
        config = get_config()[model_num]
        model_bindings = h.to_bindings(config)
        model_config_file = resources.get_file(
            "config/correlated_factors_study_ws_id2/model_configs/shared.gin")
        return model_bindings, model_config_file

    def get_postprocess_config_files(self):
        """Returns postprocessing config files."""
        return list(
            resources.get_files_in_folder(
                "config/correlated_factors_study_ws_id2/postprocess_configs/"))

    def get_eval_config_files(self):
        """Returns evaluation config files."""
        return list(
            resources.get_files_in_folder(
                "config/correlated_factors_study_ws_id2/metric_configs/"))
