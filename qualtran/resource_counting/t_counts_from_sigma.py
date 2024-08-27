#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Mapping

import cirq

from qualtran import Adjoint, Bloq, Controlled
from qualtran.symbolics import ceil, SymbolicInt


def t_count_for_gate(bloq: Bloq):
    from qualtran.resource_counting.classify_bloqs import bloq_is_clifford, bloq_is_rotation
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.bloqs.basic_gates import TGate

    if isinstance(bloq, Adjoint):
        return t_count_for_gate(bloq.subbloq)
    if isinstance(bloq, Controlled):
        return 4 + t_count_for_gate(bloq.subbloq)
    if isinstance(bloq, TGate):
        return 1
    if bloq_is_clifford(bloq):
        return 0
    if bloq_is_rotation(bloq) and not cirq.has_stabilizer_effect(bloq):
        assert hasattr(bloq, 'eps')
        return ceil(TComplexity.rotation_cost(bloq.eps))
    return bloq.t_complexity().t


def t_counts_from_sigma(sigma: Mapping['Bloq', SymbolicInt]) -> SymbolicInt:
    """Aggregates T-counts from a sigma dictionary by summing T-costs for all rotation bloqs."""

    return sum(count * t_count_for_gate(bloq) for bloq, count in sigma.items())
