# Copyright 2024 ByteDance and/or its affiliates.
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
"""Protenix selectivity-aware binder design via a gradient-free genetic
algorithm using Protenix as the structure-and-metrics oracle.

Implements the selectivity-aware design method of Pan et al. (GEM workshop,
ICLR 2026) with Protenix as the structure-and-metrics oracle.
"""

from proga.config import DesignConfig, NANOBODY_SCAFFOLD
from proga.complex import (
    DesignMask,
    PMHCTarget,
    build_sample_dict,
    build_sample_dict_n,
    load_pmhc_from_cif,
    parse_design_mask,
)
from proga.spec import (
    BinderSpec,
    FixedChain,
    OffTarget,
    TargetSpec,
    pmhc_spec_from_cifs,
)

__all__ = [
    "DesignConfig",
    "NANOBODY_SCAFFOLD",
    "DesignMask",
    "PMHCTarget",
    "build_sample_dict",
    "build_sample_dict_n",
    "load_pmhc_from_cif",
    "parse_design_mask",
    "TargetSpec",
    "BinderSpec",
    "FixedChain",
    "OffTarget",
    "pmhc_spec_from_cifs",
]
