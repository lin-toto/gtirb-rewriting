# GTIRB-Rewriting Rewriting API for GTIRB
# Copyright (C) 2021 GrammaTech, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This project is sponsored by the Office of Naval Research, One Liberty
# Center, 875 N. Randolph Street, Arlington, VA 22203 under contract #
# N68335-17-C-0700.  The content of the information does not necessarily
# reflect the position or policy of the Government and no official
# endorsement should be inferred.

import gtirb
import mcasm

_INDIRECT_CALL_INSTRS = {
    gtirb.Module.ISA.IA32: {
        "CALL32m",
        "CALL32r",
        "FARCALL16i",
        "FARCALL32i",
        "FARCALL16m",
        "FARCALL32m",
    },
    gtirb.Module.ISA.X64: {
        "CALL64m",
        "CALL64r",
        "FARCALL16m",
        "FARCALL32m",
        "FARCALL64m",
    },
    gtirb.Module.ISA.ARM64: {"BLR", "BLRAA", "BLRAAZ", "BLRAB", "BLRABZ"},
    gtirb.Module.ISA.MIPS32: {"JALR"},
    gtirb.Module.ISA.ValidButUnsupported: {"JALR", "C_JALR"},
}


def is_indirect_call(
    isa: gtirb.Module.ISA, inst: mcasm.mc.Instruction
) -> bool:
    if not inst.desc.is_call:
        return False

    if isa not in _INDIRECT_CALL_INSTRS:
        raise NotImplementedError("unknown ISA")

    # Sadly LLVM does not expose this directly, so we need to have per-ISA
    # knowledge about specific instruction forms. Plus these are _LLVM_
    # instructions, not the actual ISA's instructions.
    return inst.name in _INDIRECT_CALL_INSTRS[isa]


def riscv_edge_type(inst: mcasm.mc.Instruction):
    """Override LLVM transfer flags using the expanded register operands."""
    name, operands = inst.name, inst.operands
    if name in {"JAL", "JALR"}:
        if operands[0].name != "X0":
            return gtirb.Edge.Type.Call
        if (
            name == "JALR" and operands[1].name in {"X1", "X5"}
            and operands[2] == 0
        ):
            return gtirb.Edge.Type.Return
        return gtirb.Edge.Type.Branch
    if name in {"C_JAL", "C_JALR"}:
        return gtirb.Edge.Type.Call
    if name == "C_JR":
        return (
            gtirb.Edge.Type.Return
            if operands[0].name in {"X1", "X5"}
            else gtirb.Edge.Type.Branch
        )
    if name == "C_J":
        return gtirb.Edge.Type.Branch
    return None
