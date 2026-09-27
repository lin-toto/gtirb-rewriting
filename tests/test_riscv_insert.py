"""Insertion placement preserves call pairs and pre-memory-access ordering."""

import gtirb
import pytest
from gtirb_test_helpers import (
    add_code_block,
    add_text_section,
    create_test_module,
)

from gtirb_rewriting import Constraints, Patch, RewritingContext
from gtirb_rewriting.abi import ABI, _ABIS

A = gtirb.SymbolicExpression.Attribute
NOP = bytes.fromhex("13000000")


class NoScratchABI(ABI):
    """The location tests insert bytes, not patches needing saved registers."""

    def all_registers(self):
        return []

    def nop(self):
        return NOP

    def _create_prologue_and_epilogue(
        self, constraints, register_use, is_leaf_function
    ):
        assert not register_use.clobbered_registers
        return [], [], 0


@pytest.fixture
def context_factory(monkeypatch):
    monkeypatch.setitem(
        _ABIS,
        (gtirb.Module.ISA.ValidButUnsupported, gtirb.Module.FileFormat.ELF),
        NoScratchABI(),
    )

    def create(low=0x00070713, split=False, call=False):
        ir, module = create_test_module(
            gtirb.Module.FileFormat.ELF, gtirb.Module.ISA.ValidButUnsupported
        )
        module.aux_data["archInfo"] = gtirb.AuxData(
            {"ISA": "RISCV64"}, "mapping<string,string>"
        )
        _, interval = add_text_section(module, address=0x1000)
        words = (
            [0x00000317, 0x000300E7]
            if call else [0x00000717, 0x000002B7, 0x00028293, low]
        )
        contents = b"".join(w.to_bytes(4, "little") for w in words) + NOP
        block = add_code_block(interval, contents)
        second = block
        if split:
            block.size = 4
            second = gtirb.CodeBlock(
                size=len(contents) - 4, offset=4, byte_interval=interval
            )
            ir.cfg.add(gtirb.Edge(
                block, second, gtirb.Edge.Label(gtirb.Edge.Type.Fallthrough)
            ))
        target = gtirb.Symbol(
            "target", payload=gtirb.ProxyBlock(module=module), module=module
        )
        anchor = gtirb.Symbol("anchor", payload=block, module=module)
        interval.symbolic_expressions[0] = gtirb.SymAddrConst(
            0, target, {A.HI, A.PCREL}
        )
        if call:
            interval.symbolic_expressions[4] = gtirb.SymAddrConst(
                0, anchor, {A.LO, A.PCREL}
            )
        else:
            interval.symbolic_expressions[4] = gtirb.SymAddrConst(
                0, target, {A.HI}
            )
            interval.symbolic_expressions[8] = gtirb.SymAddrConst(
                0, target, {A.LO}
            )
            interval.symbolic_expressions[12] = gtirb.SymAddrConst(
                0, anchor, {A.LO, A.PCREL}
            )
        return RewritingContext(module, []), block, second

    return create


@pytest.mark.parametrize(
    "low, expected", [(0x00070713, 16), (0x00073703, 4), (0x00E73023, 4)]
)
def test_absolute_low_is_not_the_pcrel_partner(context_factory, low, expected):
    context, block, _ = context_factory(low=low)
    assert context.resolve_insert_location(block, 4) == (block, expected)
    context.insert_at(block=block, offset=4, patch=NOP)
    (modification,) = context._modifications._block_changes[block]
    (offset,) = modification.scope._potential_offsets(block, None)
    assert offset == expected


@pytest.mark.parametrize("split", [False, True])
def test_call_checks_stay_before_auipc(context_factory, split):
    context, block, second = context_factory(call=True, split=split)
    offset = 0 if split else 4
    assert context.resolve_insert_location(second, offset) == (block, 0)
    context.insert_at(
        second, offset,
        Patch.from_function(lambda _: ".option norvc\nnop", Constraints()),
    )
    context.apply()
    assert block.contents.startswith(NOP)
    # The surviving LO still names AUIPC, not the inserted NOP.
    low = next(
        e for bi in block.module.byte_intervals
        for e in bi.symbolic_expressions.values() if A.LO in e.attributes
    )
    anchor = low.symbol.referent
    assert anchor.byte_interval.contents[
        anchor.offset : anchor.offset + 4
    ] == bytes.fromhex("17030000")
