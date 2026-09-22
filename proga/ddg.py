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
"""Pluggable predicted differential-binding (ddG) term (eq 14).

The paper uses BA-DDG (Jiao et al., 2024) to score
``ddG_pred = dG_mutant(off-target) - dG_wildtype(on-target)``, where a positive
value indicates preferential binding to the on-target peptide. BA-DDG is not
bundled with Protenix and ships no weights here, so the default predictor is
*disabled* and contributes 0 to fitness. The genetic loop only invokes the
predictor once the generation index reaches ``DesignConfig.t_ddg``.

To enable the energetic selectivity term, implement :class:`DDGPredictor` (it
receives the designed nanobody plus the on- and off-target peptides, would fold
the off-target complex with the same oracle, and return the raw ``ddG_pred``)
and pass an instance to the genetic runner. The sigmoid scaling of eq 14 is
applied in ``scoring.py``, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class DDGPredictor(ABC):
    """Interface for a predicted differential-binding model."""

    #: Whether this predictor contributes a non-trivial term to the fitness.
    enabled: bool = True

    @abstractmethod
    def predict(
        self,
        nanobody_seq: str,
        target_peptide: str,
        off_target_peptide: str,
        mhc_seq: str,
    ) -> float:
        """Return raw ``ddG_pred`` (eq 14); positive favours the on-target."""


class DisabledDDG(DDGPredictor):
    """Default no-op predictor: returns 0 and is skipped by the fitness sum."""

    enabled = False

    def predict(
        self,
        nanobody_seq: str,
        target_peptide: str,
        off_target_peptide: str,
        mhc_seq: str,
    ) -> float:
        return 0.0


def default_ddg_predictor(predictor: Optional[DDGPredictor] = None) -> DDGPredictor:
    """Return ``predictor`` if given, else the disabled default."""
    return predictor if predictor is not None else DisabledDDG()
