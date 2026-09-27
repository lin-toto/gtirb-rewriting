"""Instruction decoding for the architectures supported by the rewriter.

GTIRB represents RISC-V through ``archInfo`` until its ISA enum grows a
RISC-V member. Keep that extension local: importing this module does not
modify gtirb-capstone's decoder or any other client's decoder configuration.
"""

import gtirb
from gtirb_capstone.capstone_compatibility import capstone
from gtirb_capstone.instructions import (
    GtirbInstructionDecoder as _GtirbInstructionDecoder,
)


RISCV64_MODE = (
    capstone.CS_MODE_RISCV64
    | capstone.CS_MODE_RISCVC
    | getattr(capstone, "CS_MODE_RISCV_A", 0)
    | getattr(capstone, "CS_MODE_RISCV_FD", 0)
)


def configure_riscv64(decoder: capstone.Cs) -> capstone.Cs:
    """Decode RV64GC with complete operands, including link registers.

    Capstone 6's alias detail omits operands of JAL/JALR/RET. Its real,
    uncompressed detail mode preserves those operands while retaining each
    instruction's encoded size. Capstone 5 already exposes the real details.
    """
    if hasattr(capstone, "CS_OPT_SYNTAX_UNCOMPRESSED_REAL"):
        decoder.syntax = capstone.CS_OPT_SYNTAX_UNCOMPRESSED_REAL
        decoder.option(
            capstone.CS_OPT_DETAIL,
            capstone.CS_OPT_ON | capstone.CS_OPT_DETAIL_UNCOMPRESSED_REAL,
        )
    else:
        decoder.detail = True
    return decoder


def riscv64_decoder(mode: int = 0) -> capstone.Cs:
    """Create an RV64GC decoder; ``mode`` may add byte-order options."""
    return configure_riscv64(
        capstone.Cs(capstone.CS_ARCH_RISCV, RISCV64_MODE | mode)
    )


def module_is_riscv64(module: gtirb.Module) -> bool:
    if module is None or "archInfo" not in module.aux_data:
        return False
    arch_info = module.aux_data["archInfo"].data
    return (
        isinstance(arch_info, dict)
        and str(arch_info.get("ISA", "")).upper() == "RISCV64"
    )


class GtirbInstructionDecoder(_GtirbInstructionDecoder):
    """gtirb-capstone's decoder extended for ``archInfo.ISA=RISCV64``."""

    def _get_block_decoder(self, block: gtirb.CodeBlock, opts: int = 0):
        if (
            self._arch == gtirb.Module.ISA.ValidButUnsupported
            and module_is_riscv64(block.module)
        ):
            endian = (
                capstone.CS_MODE_BIG_ENDIAN
                if block.module.byte_order == gtirb.Module.ByteOrder.Big
                else capstone.CS_MODE_LITTLE_ENDIAN
            )
            key = ("riscv64", endian | opts)
            if key not in self._cs:
                self._cs[key] = riscv64_decoder(endian | opts)
            return self._cs[key]
        return super()._get_block_decoder(block, opts)
