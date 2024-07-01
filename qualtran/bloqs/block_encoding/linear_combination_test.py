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

import pytest
from attrs import evolve

from qualtran import QAny, QBit, Register, Signature
from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate
from qualtran.bloqs.block_encoding.linear_combination import (
    _linear_combination_block_encoding,
    LinearCombination,
)
from qualtran.bloqs.block_encoding.unitary import Unitary


def test_linear_combination(bloq_autotester):
    bloq_autotester(_linear_combination_block_encoding)


def test_linear_combination_signature():
    assert _linear_combination_block_encoding().signature == Signature(
        [Register("system", QBit()), Register("ancilla", QAny(1)), Register("resource", QAny(0))]
    )
    with pytest.raises(ValueError):
        _ = LinearCombination([], [])
    with pytest.raises(ValueError):
        _ = LinearCombination([Unitary(TGate(), dtype=QBit())], [])
    with pytest.raises(ValueError):
        _ = LinearCombination(
            [Unitary(TGate(), dtype=QBit()), Unitary(CNOT(), dtype=QAny(2))], [1.0]
        )


def test_linear_combination_params():
    u1 = evolve(
        Unitary(TGate(), dtype=QBit()), alpha=0.5, num_ancillas=2, num_resource=1, epsilon=0.01
    )
    u2 = evolve(
        Unitary(Hadamard(), dtype=QBit()), alpha=0.5, num_ancillas=1, num_resource=1, epsilon=0.1
    )
    bloq = LinearCombination([u1, u2], [0.5, -0.5])
    assert bloq.dtype == QBit()
    assert bloq.alpha == 0.5 * 0.5 + 0.5 * 0.5
    assert bloq.epsilon == (0.5 + 0.5) * max(0.01, 0.1)
    assert bloq.num_ancillas == 1 + max(1, 2)
    assert bloq.num_resource == 1 + 1
