"""Keep frontend liveness attached to surviving instructions, not positions."""

import io

import gtirb
import pytest
from gtirb_test_helpers import (
    add_code_block,
    add_text_section,
    create_test_module,
)
from helpers import literal_patch

from gtirb_rewriting import RewritingContext
from gtirb_rewriting import _auxdata


def make_module():
    ir, module = create_test_module(
        gtirb.Module.FileFormat.ELF, gtirb.Module.ISA.X64
    )
    _, interval = add_text_section(module, address=0x1000)
    block = add_code_block(interval, b"\x90\x90\x90\xC3")
    _auxdata.live_register_sets.set(
        module, {gtirb.Offset(block, i): 1 << i for i in range(4)}
    )
    _auxdata.live_register_sets_high.set(
        module, {gtirb.Offset(block, i): (1 << i) + 100 for i in range(4)}
    )
    return ir, module, interval, block


def masks_by_position(module):
    result = {}
    low = _auxdata.live_register_sets.get(module)
    high = _auxdata.live_register_sets_high.get(module)
    assert dict(high) == {offset: mask + 100 for offset, mask in low.items()}
    for offset, mask in _auxdata.live_register_sets.get(module).items():
        block = offset.element_id
        assert isinstance(block, gtirb.CodeBlock)
        assert block.module is module
        assert 0 <= offset.displacement < block.size
        result[block.offset + offset.displacement] = mask
    return result


def test_live_register_insertions_across_rounds():
    ir, module, interval, block = make_module()
    ctx = RewritingContext(module, [])
    ctx.insert_at(block, 0, literal_patch("nop"))
    ctx.insert_at(block, 2, literal_patch("nop; nop"))
    ctx.apply()
    assert masks_by_position(module) == {1: 1, 2: 2, 5: 4, 6: 8}

    # Serialization resolves offset UUIDs back to the current block objects.
    stream = io.BytesIO()
    ir.save_protobuf_file(stream)
    stream.seek(0)
    ir = gtirb.IR.load_protobuf_file(stream)
    module = ir.modules[0]
    offset = next(
        offset for offset, mask in _auxdata.live_register_sets.get(module).items()
        if mask == 4
    )
    ctx = RewritingContext(module, [])
    ctx.insert_at(offset.element_id, offset.displacement, literal_patch("nop"))
    ctx.apply()
    assert masks_by_position(module) == {1: 1, 2: 2, 6: 4, 7: 8}


@pytest.mark.parametrize("replacement", ["nop", "nop; nop; nop"])
def test_live_register_replacements(replacement):
    _, module, interval, block = make_module()
    ctx = RewritingContext(module, [])
    ctx.replace_at(block, 1, 1, literal_patch(replacement))
    ctx.apply()
    extra = interval.size - 4
    assert masks_by_position(module) == {0: 1, 2 + extra: 4, 3 + extra: 8}


def test_live_register_split_and_delete():
    _, module, interval, block = make_module()
    ctx = RewritingContext(module, [])
    ctx.insert_at(block, 1, literal_patch("jmp .L_resume; .L_resume:"))
    ctx.apply()
    added = interval.size - 4
    assert len(tuple(module.code_blocks)) > 1
    assert masks_by_position(module) == {
        0: 1, 1 + added: 2, 2 + added: 4, 3 + added: 8,
    }

    offset = next(
        offset for offset, mask in _auxdata.live_register_sets.get(module).items()
        if mask == 2
    )
    ctx = RewritingContext(module, [])
    ctx.delete_at(offset.element_id, offset.displacement, 1)
    ctx.apply()
    assert masks_by_position(module) == {0: 1, 1 + added: 4, 2 + added: 8}


def test_live_register_delete_whole_block():
    _, module, _, block = make_module()
    ctx = RewritingContext(module, [])
    ctx.delete_at(block, 0, block.size)
    ctx.apply()
    assert masks_by_position(module) == {}


def test_resized_overlapping_aliases_invalidate_liveness():
    _, module, interval, primary = make_module()
    aliases = [
        gtirb.CodeBlock(offset=0, size=4, byte_interval=interval),
        gtirb.CodeBlock(offset=1, size=3, byte_interval=interval),
    ]
    for table in (_auxdata.live_register_sets, _auxdata.live_register_sets_high):
        facts = table.get(module)
        for alias in aliases:
            facts[gtirb.Offset(alias, 0)] = 7
            facts[gtirb.Offset(alias, alias.size - 1)] = 8

    ctx = RewritingContext(module, [])
    ctx.insert_at(primary, 2, literal_patch("nop"))
    ctx.apply()

    assert [bytes(alias.contents) for alias in aliases] == [
        b"\x90\x90\x90\x90\xc3", b"\x90\x90\x90\xc3",
    ]
    for table in (_auxdata.live_register_sets, _auxdata.live_register_sets_high):
        # Missing alias facts tell the consumer to assume every register live.
        assert all(
            offset.element_id not in aliases for offset in table.get(module)
        )
    assert masks_by_position(module) == {0: 1, 1: 2, 3: 4, 4: 8}


def test_same_size_replacement_invalidates_overlapping_alias_liveness():
    _, module, interval, primary = make_module()
    aliases = [
        gtirb.CodeBlock(offset=0, size=4, byte_interval=interval),
        gtirb.CodeBlock(offset=1, size=3, byte_interval=interval),
    ]
    for table in (_auxdata.live_register_sets, _auxdata.live_register_sets_high):
        facts = table.get(module)
        for alias in aliases:
            facts[gtirb.Offset(alias, 0)] = 7
            facts[gtirb.Offset(alias, alias.size - 1)] = 8

    ctx = RewritingContext(module, [])
    ctx.replace_at(primary, 1, 2, literal_patch("xorl %eax, %eax"))
    ctx.apply()

    assert interval.size == 4
    assert [bytes(alias.contents) for alias in aliases] == [
        b"\x90\x31\xc0\xc3", b"\x31\xc0\xc3",
    ]
    for table in (_auxdata.live_register_sets, _auxdata.live_register_sets_high):
        assert all(
            offset.element_id not in aliases for offset in table.get(module)
        )
    assert masks_by_position(module) == {0: 1, 3: 8}
