"""PC-relative low relocations follow instructions, not entry-point symbols."""

import io

import gtirb
import pytest
from gtirb_test_helpers import add_code_block, add_text_section, create_test_module

from gtirb_rewriting._modify.edit import edit_byte_interval
from gtirb_rewriting.prepare import prepare_for_rewriting


NOP = bytes.fromhex("13000000")
HI = bytes.fromhex("97020000")  # auipc t0,0
LO = bytes.fromhex("93820200")  # addi t0,t0,0
A = gtirb.SymbolicExpression.Attribute


def make_pair(bits=64, got=False):
    ir, module = create_test_module(
        gtirb.Module.FileFormat.ELF, gtirb.Module.ISA.ValidButUnsupported
    )
    module.aux_data["archInfo"] = gtirb.AuxData(
        {"ISA": f"RISCV{bits}"}, "mapping<string,string>"
    )
    _, interval = add_text_section(module, address=0x1000)
    block = add_code_block(interval, HI + LO + NOP)
    entry = gtirb.Symbol("entry", payload=block, module=module)
    target = gtirb.Symbol("target", payload=gtirb.ProxyBlock(module=module), module=module)
    interval.symbolic_expressions[0] = gtirb.SymAddrConst(
        0, target, {A.GOT} if got else {A.HI, A.PCREL}
    )
    interval.symbolic_expressions[4] = gtirb.SymAddrConst(0, entry, {A.LO, A.PCREL})
    return ir, module, block, entry


def low_location(module):
    return next(
        (interval, offset, expression)
        for interval in module.byte_intervals
        for offset, expression in interval.symbolic_expressions.items()
        if A.LO in expression.attributes
    )


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("got", [False, True])
def test_pcrel_anchor_moves_but_entry_stays(bits, got):
    ir, module, block, entry = make_pair(bits, got)
    for _ in range(3):
        with prepare_for_rewriting(module, NOP):
            interval = block.byte_interval
            edit_byte_interval(interval, block.offset, 0, NOP, (block,))
            block.size += len(NOP)
        interval, offset, low = low_location(module)
        assert low.symbol is not entry
        assert low.symbol.referent.byte_interval is interval
        assert low.symbol.referent.offset == offset - 4
        assert interval.contents[entry.referent.offset:entry.referent.offset + 4] == NOP
        stream = io.BytesIO()
        ir.save_protobuf_file(stream)
        stream.seek(0)
        ir = gtirb.IR.load_protobuf_file(stream)
        module = ir.modules[0]
        entry = next(module.symbols_named("entry"))
        block = entry.referent


def test_new_auipc_at_old_anchor_does_not_steal_pair():
    _, module, block, entry = make_pair()
    with prepare_for_rewriting(module, NOP):
        interval = block.byte_interval
        edit_byte_interval(interval, block.offset, 0, HI, (block,))
        block.size += 4
        decoy = gtirb.Symbol("decoy", payload=gtirb.ProxyBlock(module=module), module=module)
        interval.symbolic_expressions[block.offset] = gtirb.SymAddrConst(
            0, decoy, {A.HI, A.PCREL}
        )
    interval, offset, low = low_location(module)
    assert low.symbol.referent.offset == offset - 4
    assert low.symbol.referent.offset != entry.referent.offset
    assert interval.symbolic_expressions[low.symbol.referent.offset].symbol.name == "target"


def test_cross_interval_lo_and_hi_move_independently():
    _, module, block, entry = make_pair()
    high_interval = block.byte_interval
    low = high_interval.symbolic_expressions.pop(4)
    high_interval.contents = HI
    high_interval.size = block.size = 4
    low_interval = gtirb.ByteInterval(
        address=0x2000, contents=LO + NOP, section=high_interval.section,
        symbolic_expressions={0: low},
    )
    low_block = gtirb.CodeBlock(size=8, byte_interval=low_interval)
    with prepare_for_rewriting(module, NOP):
        edit_byte_interval(block.byte_interval, block.offset, 0, NOP, (block,))
        block.size += 4
        edit_byte_interval(low_block.byte_interval, low_block.offset, 0, NOP * 3, (low_block,))
        low_block.size += 12
    interval, offset, low = low_location(module)
    assert offset == 12
    assert low.symbol.referent.byte_interval is not interval
    assert low.symbol.referent.offset == entry.referent.offset + 4


def test_multiple_consumers_keep_the_same_anchor():
    _, module, block, _ = make_pair()
    interval = block.byte_interval
    low = interval.symbolic_expressions[4]
    # Both loads use t0 without overwriting it.
    interval.contents[4:] = bytes.fromhex("03a50200 83a50200")
    interval.symbolic_expressions[8] = gtirb.SymAddrConst(0, low.symbol, low.attributes)
    with prepare_for_rewriting(module, NOP):
        edit_byte_interval(block.byte_interval, block.offset, 0, NOP, (block,))
        block.size += 4
    lows = [expression for interval in module.byte_intervals
            for expression in interval.symbolic_expressions.values()
            if A.LO in expression.attributes]
    assert len(lows) == 2
    assert lows[0].symbol is lows[1].symbol


@pytest.mark.parametrize("replacement", [b"", NOP, HI])
def test_deleted_hi_with_surviving_lo_is_rejected(replacement):
    _, module, block, _ = make_pair()
    with pytest.raises(ValueError, match="PC-relative.*HI.*removed"):
        with prepare_for_rewriting(module, NOP):
            edit_byte_interval(block.byte_interval, block.offset, 4, replacement)


def test_removing_both_endpoints_is_valid():
    _, module, block, _ = make_pair()
    with prepare_for_rewriting(module, NOP):
        edit_byte_interval(block.byte_interval, block.offset, 8, b"")
    assert not any(interval.symbolic_expressions for interval in module.byte_intervals)


def test_editing_encoded_hi_in_place_is_rejected():
    _, module, block, _ = make_pair()
    with pytest.raises(ValueError, match="PC-relative.*instruction.*changed"):
        with prepare_for_rewriting(module, NOP):
            interval = block.byte_interval
            interval.contents[block.offset:block.offset + 4] = bytes.fromhex("17030000")


@pytest.mark.parametrize("bits", [32, 64])
@pytest.mark.parametrize("prefix", [NOP, HI])
def test_replaced_lo_cannot_keep_a_stale_public_anchor(bits, prefix):
    _, module, block, _ = make_pair(bits)
    with pytest.raises(ValueError, match="PC-relative.*(anchor|replaced)"):
        with prepare_for_rewriting(module, NOP):
            interval = block.byte_interval
            edit_byte_interval(interval, block.offset, 0, prefix, (block,))
            block.size += 4
            offset = block.offset + 8
            low = interval.symbolic_expressions[offset]
            interval.symbolic_expressions[offset] = gtirb.SymAddrConst(
                low.offset, low.symbol, low.attributes
            )
            if prefix == HI:
                # Even an identical AUIPC at the old entry is not this pair's
                # original producer. Structural validation alone is insufficient.
                high = interval.symbolic_expressions[block.offset + 4]
                interval.symbolic_expressions[block.offset] = gtirb.SymAddrConst(
                    high.offset, high.symbol, high.attributes
                )


@pytest.mark.parametrize("new_anchor", [False, True])
def test_replaced_lo_with_a_valid_anchor_is_accepted(new_anchor):
    _, module, block, entry = make_pair()
    with prepare_for_rewriting(module, NOP):
        interval = block.byte_interval
        offset = block.offset + 4
        low = interval.symbolic_expressions[offset]
        if new_anchor:
            edit_byte_interval(interval, block.offset, 0, NOP, (block,))
            block.size += 4
            offset += 4
            anchor = gtirb.Symbol("replacement_anchor", payload=gtirb.CodeBlock(
                size=0, offset=block.offset + 4, byte_interval=interval
            ), module=module)
        else:
            anchor = entry
        interval.symbolic_expressions[offset] = gtirb.SymAddrConst(
            low.offset, anchor, low.attributes
        )
    interval, offset, low = low_location(module)
    assert low.symbol is anchor
    assert low.symbol.referent.offset == offset - 4


def test_replaced_lo_cannot_hide_replacement_of_its_hi():
    _, module, block, _ = make_pair()
    with pytest.raises(ValueError, match="PC-relative.*replaced"):
        with prepare_for_rewriting(module, NOP):
            interval = block.byte_interval
            for offset in (block.offset, block.offset + 4):
                expression = interval.symbolic_expressions[offset]
                interval.symbolic_expressions[offset] = gtirb.SymAddrConst(
                    expression.offset, expression.symbol, expression.attributes
                )


def test_replacing_both_endpoints_with_an_explicit_anchor_is_valid():
    _, module, block, _ = make_pair()
    with prepare_for_rewriting(module, NOP):
        interval = block.byte_interval
        high = interval.symbolic_expressions[block.offset]
        anchor = gtirb.Symbol("new_producer", payload=gtirb.CodeBlock(
            size=0, offset=block.offset, byte_interval=interval
        ), module=module)
        interval.symbolic_expressions[block.offset] = gtirb.SymAddrConst(
            high.offset, high.symbol, high.attributes
        )
        interval.symbolic_expressions[block.offset + 4] = gtirb.SymAddrConst(
            0, anchor, {A.LO, A.PCREL}
        )
    assert low_location(module)[2].symbol is anchor


def test_replaced_consumer_of_a_repaired_private_anchor_is_valid():
    _, module, block, _ = make_pair()
    interval = block.byte_interval
    low = interval.symbolic_expressions[4]
    interval.contents[4:] = bytes.fromhex("03a50200 83a50200")
    interval.symbolic_expressions[8] = gtirb.SymAddrConst(
        0, low.symbol, low.attributes
    )
    with prepare_for_rewriting(module, NOP):
        pass
    with prepare_for_rewriting(module, NOP):
        interval = block.byte_interval
        edit_byte_interval(interval, block.offset, 0, NOP, (block,))
        block.size += 4
        low = interval.symbolic_expressions[block.offset + 8]
        interval.symbolic_expressions[block.offset + 8] = gtirb.SymAddrConst(
            low.offset, low.symbol, low.attributes
        )
    lows = [expression for interval in module.byte_intervals
            for expression in interval.symbolic_expressions.values()
            if A.LO in expression.attributes]
    assert len(lows) == 2
    assert lows[0].symbol is lows[1].symbol
    assert lows[0].symbol.referent.offset == block.offset + 4
