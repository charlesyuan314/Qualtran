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

import math
from functools import cached_property
from typing import Dict, Sequence, Tuple

from attrs import field, frozen, validators

from qualtran import (
    bloq_example,
    BloqBuilder,
    BloqDocSpec,
    QAny,
    QDType,
    Register,
    Signature,
    Soquet,
    SoquetT,
)
from qualtran.bloqs.block_encoding import BlockEncoding
from qualtran.bloqs.block_encoding.lcu_select_and_prepare import PrepareOracle
from qualtran.symbolics import SymbolicFloat


@frozen
class LinearCombination(BlockEncoding):
    r"""Linear combination of a sequence of block encodings.

    Builds the block encoding $B[\lambda_1 U_1 + \lambda_2 U_2 + \cdots + \lambda_n U_n]$ given block encodings $B[U_1], \ldots, B[U_n]$ and coefficients $\lambda_i \in \mathbb{R}$.

    When each $B[U_i]$ is a $(\alpha_i, a_i, \epsilon_i)$-block encoding of $U_i$, we have that
    $B[\lambda_1 U_1 + \cdots + \lambda_n U_n]$ is a $(\sum_i \lvert\lambda_i\rvert\alpha_i, \lceil \log_2 n \rceil + \max_i a_i, (\sum_i \lvert\lambda_i\rvert)\max_i \epsilon_i)$-block encoding of $\lambda_1 U_1 + \cdots + \lambda_n U_n$.

    Args:
        U: A sequence of block encodings.

    Registers:
        system: The system register.
        ancilla: The ancilla register.
        resource: The resource register.

    References:
        [Quantum algorithms: A survey of applications and end-to-end complexities](https://arxiv.org/abs/2310.03011). Dalzell et al. (2023). Ch. 10.2.
    """

    U: Sequence[BlockEncoding] = field(
        converter=lambda x: x if isinstance(x, tuple) else tuple(x), validator=validators.min_len(1)
    )
    lambd: Sequence[float] = field(converter=lambda x: x if isinstance(x, tuple) else tuple(x))

    def __attrs_post_init__(self):
        if len(self.U) != len(self.lambd):
            raise ValueError("Must provide the same number of block encodings and coefficients.")
        if not all(u.dtype == self.dtype for u in self.U):
            raise ValueError("All block encodings must have the same dtype.")

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            system=self.dtype, ancilla=QAny(self.num_ancillas), resource=QAny(self.num_resource)
        )

    @cached_property
    def dtype(self) -> QDType:
        return self.U[0].dtype

    def pretty_name(self) -> str:
        return f"B[{'+'.join(u.pretty_name()[2:-1] for u in self.U)}]"

    @cached_property
    def alpha(self) -> SymbolicFloat:
        return sum(abs(l) * u.alpha for u, l in zip(self.U, self.lambd))

    @cached_property
    def num_ancillas(self) -> int:
        return max(u.num_ancillas for u in self.U) + math.ceil(math.log2(len(self.U)))

    @cached_property
    def num_resource(self) -> int:
        return sum(u.num_resource for u in self.U)

    @cached_property
    def epsilon(self) -> SymbolicFloat:
        return sum(abs(l) for l in self.lambd) * max(u.epsilon for u in self.U)

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
        return {"system": system, "ancilla": ancilla, "resource": resource}


@bloq_example
def _linear_combination_block_encoding() -> LinearCombination:
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    linear_combination_block_encoding = LinearCombination(
        [Unitary(TGate(), dtype=QBit()), Unitary(Hadamard(), dtype=QBit())], [0.5, 0.5]
    )
    return linear_combination_block_encoding


_LINEAR_COMBINATION_DOC = BloqDocSpec(
    bloq_cls=LinearCombination,
    import_line="from qualtran.bloqs.block_encoding import LinearCombination",
    examples=[_linear_combination_block_encoding],
)
