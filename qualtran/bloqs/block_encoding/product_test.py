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

import numpy as np
import pytest
from attrs import evolve

from qualtran import BloqBuilder, QAny, QBit, Register, Signature, Soquet
from qualtran.bloqs.basic_gates import CNOT, Hadamard, TGate, ZeroEffect, ZeroState
from qualtran.bloqs.block_encoding.product import _product_block_encoding, Product
from qualtran.bloqs.block_encoding.unitary import Unitary


def test_product(bloq_autotester):
    bloq_autotester(_product_block_encoding)


def test_product_signature():
    assert _product_block_encoding().signature == Signature(
        [Register("system", QBit()), Register("ancilla", QAny(1)), Register("resource", QAny(0))]
    )
    with pytest.raises(AssertionError):
        _ = Product([])
    with pytest.raises(AssertionError):
        _ = Product([Unitary(TGate(), dtype=QBit()), Unitary(CNOT(), dtype=QAny(2))])


def test_product_params():
    u1 = evolve(
        Unitary(TGate(), dtype=QBit()), alpha=0.5, num_ancillas=2, num_resource=1, epsilon=0.01
    )
    u2 = evolve(
        Unitary(Hadamard(), dtype=QBit()), alpha=0.5, num_ancillas=1, num_resource=1, epsilon=0.1
    )
    bloq = Product([u1, u2])
    assert bloq.dtype == QBit()
    assert bloq.alpha == 0.5 * 0.5
    assert bloq.epsilon == 0.5 * 0.01 + 0.5 * 0.1
    assert bloq.num_ancillas == max(2, 1) + 1
    assert bloq.num_resource == 1 + 1


def test_product_tensors():
    bb = BloqBuilder()
    system = bb.add_register("system", 1)
    ancilla: Soquet = bb.add(ZeroState())
    resource = bb.add_register("resource", 0)
    system, ancilla, resource = bb.add_t(
        _product_block_encoding(), system=system, ancilla=ancilla, resource=resource
    )
    bb.add(ZeroEffect(), q=ancilla)
    bloq = bb.finalize(system=system, resource=resource)

    from_gate = np.matmul(TGate().tensor_contract(), Hadamard().tensor_contract())
    from_tensors = bloq.tensor_contract()
    np.testing.assert_allclose(from_gate, from_tensors)
