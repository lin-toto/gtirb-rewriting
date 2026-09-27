"""The public decoder handles RV64 without changing gtirb-capstone."""

import logging

import gtirb
import pytest
from gtirb_capstone.instructions import GtirbInstructionDecoder as Upstream
from gtirb_test_helpers import (
    add_code_block,
    add_text_section,
    create_test_module,
)

from gtirb_rewriting.decoder import GtirbInstructionDecoder, riscv64_decoder
from gtirb_rewriting.utils import show_block_asm


def make_block(contents, isa="RISCV64"):
    _, module = create_test_module(
        gtirb.Module.FileFormat.ELF, gtirb.Module.ISA.ValidButUnsupported
    )
    module.aux_data["archInfo"] = gtirb.AuxData(
        {"ISA": isa}, "mapping<string,string>"
    )
    _, interval = add_text_section(module, address=0x1000)
    return add_code_block(interval, bytes.fromhex(contents))


def test_rv64gc_and_link_register_details(caplog):
    # c.nop; amoadd.d a0,a1,(a2); fld fa0,0(a0); jalr ra,0(t1); c.jr ra
    block = make_block("0100 2f35b600 07350500 e7000300 8280")
    upstream_method = Upstream._get_block_decoder
    decoder = GtirbInstructionDecoder(block.module.isa)
    instructions = list(decoder.get_instructions(block))
    assert [inst.size for inst in instructions] == [2, 4, 4, 4, 2]
    assert instructions[3].reg_name(instructions[3].operands[0].reg) == "ra"
    assert instructions[4].reg_name(instructions[4].operands[0].reg) == "zero"
    assert list(riscv64_decoder().disasm(block.contents, 0x1000))
    assert Upstream._get_block_decoder is upstream_method
    assert not hasattr(Upstream, "_teapot_riscv64_compat")
    with caplog.at_level(logging.DEBUG):
        show_block_asm(block)
    assert "<incomplete disassembly>" not in caplog.text


def test_unknown_unsupported_isa_is_not_guessed_as_rv64():
    block = make_block("13000000", "RISCV32")
    with pytest.raises(KeyError):
        list(GtirbInstructionDecoder(block.module.isa).get_instructions(block))
