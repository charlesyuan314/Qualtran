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

from functools import cached_property, reduce
from typing import Dict, Sequence, Tuple

from attrs import evolve, field, frozen

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QBit,
    QDType,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran._infra.controlled import CtrlSpec
from qualtran.bloqs.basic_gates.x_basis import XGate
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.bloqs.bookkeeping.partition import Partition
from qualtran.symbolics import SymbolicFloat


@frozen
class Product(BlockEncoding):
    r"""Product of a sequence of block encodings.

    Builds the block encoding $B[U_1 * U_2 * \cdots * U_n]$ given block encodings $B[U_1], \ldots, B[U_n]$.

    When each $B[U_i]$ is a $(\alpha_i, a_i, \epsilon_i)$-block encoding of $U_i$, we have that
    $B[U_1 * \cdots * U_n]$ is a $(\prod_i \alpha_i, n - 1 + \max_i a_i, \sum_i \alpha_i \epsilon_i)$-block
    encoding of $U_1 * \cdots * U_n$.

    Args:
        U: A sequence of block encodings.

    Registers:
        system: The system register.
        ancilla: The ancilla register.
        resource: The resource register.

    References:
        [Quantum algorithms: A survey of applications and end-to-end complexities](https://arxiv.org/abs/2310.03011). Dalzell et al. (2023). Ch. 10.2.
    """

    U: Sequence[BlockEncoding] = field(converter=lambda x: x if isinstance(x, tuple) else tuple(x))

    def __attrs_post_init__(self):
        assert len(self.U) > 0
        assert all(u.dtype == self.dtype for u in self.U)

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=self.dtype, ancilla=QAny(self.num_ancillas), resource=QAny(self.num_resource)
        )

    @cached_property
    def dtype(self) -> QDType:
        return self.U[0].dtype

    def pretty_name(self) -> str:
        return f"B[{'*'.join(u.pretty_name()[2:-1] for u in self.U)}]"

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return reduce(lambda a, b: a * b.alpha, self.U, 1.0)

    @cached_property
    def num_ancillas(self) -> int:
        return max(u.num_ancillas for u in self.U) + len(self.U) - 1

    @cached_property
    def num_resource(self) -> int:
        return sum(u.num_resource for u in self.U)

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return sum(u.alpha * u.epsilon for u in self.U)

    @property
    def target_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("system"),)

    @property
    def junk_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("resource"),)

    @property
    def selection_registers(self) -> Tuple[Register, ...]:
        return (self.signature.get_right("ancilla"),)

    @property
    def signal_state(self) -> PrepareOracle:
        raise NotImplementedError

    def build_composite_bloq(
        self, bb: BloqBuilder, system: SoquetT, ancilla: Soquet, resource: Soquet
    ) -> Dict[str, SoquetT]:
        # split ancilla into inner ancilla and n - 1 flags
        n = len(self.U)
        res_bits_used = 0
        for i, u in enumerate(reversed(self.U)):
            anc_part = Partition(
                self.num_ancillas,
                (
                    Register("flag_bits", dtype=QBit(), shape=(n - 1,)),  # type: ignore
                    Register("anc_used", dtype=QAny(u.num_ancillas)),
                    Register(
                        "anc_unused", dtype=QAny(self.num_ancillas - (n - 1) - u.num_ancillas)
                    ),
                ),
            )
            flag_bits, anc_used, anc_unused = bb.add_t(anc_part, x=ancilla)
            res_part = Partition(
                self.num_resource,
                (
                    Register("res_before", dtype=QAny(res_bits_used)),
                    Register("res", dtype=QAny(u.num_resource)),
                    Register(
                        "res_after", dtype=QAny(self.num_resource - res_bits_used - u.num_resource)
                    ),
                ),
            )
            res_before, res, res_after = bb.add_t(res_part, x=resource)
            res_bits_used += u.num_resource

            system, anc_used, res = bb.add_t(u, system=system, ancilla=anc_used, resource=res)
            resource = bb.add(
                evolve(res_part, partition=False),
                res_before=res_before,
                res=res,
                res_after=res_after,
            )  # type: ignore
            anc_used = bb.split(anc_used)  # type: ignore
            if i < n - 1:
                # set corresponding flag if ancillas are all zero
                anc_used, flag_bits[i] = bb.add_t(  # type: ignore
                    XGate().controlled(CtrlSpec(cvs=[0] * len(anc_used))),  # type: ignore
                    ctrl=anc_used,
                    q=flag_bits[i],  # type: ignore
                )
                flag_bits[i] = bb.add(XGate(), q=flag_bits[i])  # type: ignore
            anc_used = bb.join(anc_used)  # type: ignore
            ancilla = bb.add(
                evolve(anc_part, partition=False),
                flag_bits=flag_bits,
                anc_used=anc_used,
                anc_unused=anc_unused,
            )  # type: ignore
        return {"system": system, "ancilla": ancilla, "resource": resource}


@bloq_example
def _product_block_encoding() -> Product:
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    product_block_encoding = Product(
        [Unitary(TGate(), dtype=QBit()), Unitary(Hadamard(), dtype=QBit())]
    )
    return product_block_encoding


_PRODUCT_DOC = BloqDocSpec(
    bloq_cls=Product,
    import_line="from qualtran.bloqs.block_encoding import Product",
    examples=[_product_block_encoding],
)
