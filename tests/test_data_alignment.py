# GTIRB-Rewriting Rewriting API for GTIRB
# Copyright (C) 2026 GrammaTech, Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import gtirb
import pytest
from gtirb_test_helpers import (
    add_code_block,
    add_data_block,
    add_text_section,
    create_test_module,
)
from helpers import literal_patch

import gtirb_rewriting.prepare
from gtirb_rewriting import RewritingContext, join_byte_intervals
from gtirb_rewriting.intervalutils import PaddingError


def data_run(*, code_size=32, head_size=128, gap=0, head_alignment=None):
    _, module = create_test_module(
        gtirb.Module.FileFormat.ELF, gtirb.Module.ISA.X64
    )
    _, interval = add_text_section(module, address=0x1000)
    code = add_code_block(interval, b"\x90" * code_size)
    head_bytes = bytes(i % 256 for i in range(head_size))
    head = add_data_block(interval, head_bytes)
    interval.contents += b"\xff" * gap
    interval.size += gap
    tail = add_data_block(interval, bytes(range(160, 192)))
    alignment = {tail: 32}
    if head_alignment is not None:
        alignment[head] = head_alignment
    module.aux_data["alignment"] = gtirb.AuxData(
        alignment, "mapping<UUID,uint64_t>"
    )
    return module, code, head, tail


@pytest.mark.parametrize(
    "code_size,head_size,head_alignment",
    [(32, 128, None), (32, 128, 16), (16, 48, None)],
)
def test_code_insertion_preserves_data_run(
    code_size, head_size, head_alignment
):
    module, code, head, tail = data_run(
        code_size=code_size,
        head_size=head_size,
        head_alignment=head_alignment,
    )
    original_table = bytes(head.contents + tail.contents)
    original_alignment = dict(module.aux_data["alignment"].data)
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    ctx.apply()

    # A consumer indexing from the head still sees the exact original table,
    # even if only a later block requires alignment or the head is unaligned.
    assert head.address + head.size == tail.address
    assert bytes(
        head.byte_interval.contents[
            head.offset : head.offset + len(original_table)
        ]
    ) == original_table
    for block, boundary in original_alignment.items():
        assert block.address % boundary == 0
    assert module.aux_data["alignment"].data == original_alignment
    if code_size == 16:
        assert head.address % 32 == 16


def test_code_insertion_does_not_treat_gap_as_table_contiguity():
    module, code, head, tail = data_run(head_size=112, gap=16)
    original_head = bytes(head.contents)
    original_tail = bytes(tail.contents)
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    ctx.apply()

    assert head.address == code.address + code.size + 16
    assert tail.address > head.address + head.size + 16
    assert tail.address % 32 == 0
    assert bytes(head.contents) == original_head
    assert bytes(tail.contents) == original_tail
    assert head not in module.aux_data["alignment"].data


def test_code_insertion_does_not_infer_through_overlapping_data():
    module, code, head, tail = data_run()
    alias = gtirb.DataBlock(
        offset=head.offset + 16,
        size=16,
        byte_interval=head.byte_interval,
    )
    original_alias = bytes(alias.contents)
    original_head = bytes(head.contents)
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    ctx.apply()

    assert alias.address == head.address + 16
    assert bytes(alias.contents) == original_alias
    assert bytes(head.contents) == original_head
    assert tail.address % 32 == 0
    assert head not in module.aux_data["alignment"].data


def test_explicit_data_edit_remains_independently_alignable():
    module, code, head, tail = data_run(code_size=16, head_size=48)
    original_head = bytes(head.contents)
    original_tail = bytes(tail.contents)
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    ctx.insert_at(head, head.size, b"!")
    ctx.apply()

    assert bytes(head.contents) == original_head + b"!"
    assert bytes(tail.contents) == original_tail
    assert tail.address % 32 == 0


def test_code_insertion_rejects_inconsistent_data_alignment():
    module, code, head, _ = data_run(code_size=16, head_size=48)
    # Head and tail are separated by 48 bytes: they cannot both be 32-aligned
    # without changing the table's contents or internal offsets.
    module.aux_data["alignment"].data[head] = 32
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    with pytest.raises(PaddingError, match="incompatible alignment"):
        ctx.apply()


def test_data_alignment_survives_final_module_relayout(monkeypatch):
    module, code, head, tail = data_run(head_alignment=16)
    interval = head.byte_interval
    prefix = gtirb.CodeBlock(size=8)
    gtirb.ByteInterval(
        address=interval.address - 8,
        contents=b"\x90" * 8,
        blocks=[prefix],
        section=interval.section,
    )
    module.ir.cfg.add(
        gtirb.Edge(
            prefix, code, gtirb.Edge.Label(type=gtirb.Edge.Type.Fallthrough)
        )
    )
    # Growing the code overlaps this next interval and forces module layout.
    gtirb.ByteInterval(
        address=interval.address + interval.size,
        contents=b"suffix!!",
        blocks=[gtirb.DataBlock(size=8)],
        section=interval.section,
    )
    original_table = bytes(head.contents + tail.contents)
    original_alignment = dict(module.aux_data["alignment"].data)
    layout_calls = []
    original_layout = gtirb_rewriting.prepare.layout_module

    def layout(module):
        original_layout(module)
        layout_calls.append((head.address, tail.address))

    monkeypatch.setattr(gtirb_rewriting.prepare, "layout_module", layout)
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    ctx.apply()

    assert layout_calls
    assert head.address % 16 == 0
    assert tail.address % 32 == 0
    assert tail.address == head.address + head.size
    assert bytes(
        interval.contents[head.offset : head.offset + len(original_table)]
    ) == original_table
    assert module.aux_data["alignment"].data == original_alignment


def test_initial_layout_honors_interval_alignment_without_leaking_anchors():
    module, code, head, tail = data_run()
    interval = head.byte_interval
    interval.address = None
    alignment = module.aux_data["alignment"].data
    alignment[interval] = 64
    original_alignment = dict(alignment)
    original_data_blocks = set(module.data_blocks)
    ctx = RewritingContext(module, [])
    ctx.insert_at(code, code.size, literal_patch("nop\n" * 16))
    ctx.apply()

    assert interval.address % 64 == 0
    assert tail.address % 32 == 0
    assert head.address + head.size == tail.address
    assert set(module.data_blocks) == original_data_blocks
    assert module.aux_data["alignment"].data is alignment
    assert alignment == original_alignment


def test_join_honors_all_block_and_interval_alignment_constraints():
    first = gtirb.ByteInterval(address=0x1000, contents=b"!")
    head = gtirb.DataBlock(offset=0, size=16)
    tail = gtirb.DataBlock(offset=16, size=16)
    second = gtirb.ByteInterval(contents=b"x" * 32, blocks=[head, tail])

    joined = join_byte_intervals(
        [first, second], alignment={second: 8, head: 8, tail: 32}
    )

    assert head.address % 8 == 0
    assert tail.address % 32 == 0
    assert tail.address == head.address + 16
    assert bytes(joined.contents[head.offset : tail.offset + 16]) == b"x" * 32


def test_join_rejects_conflicting_interval_alignment_without_partial_move():
    first = gtirb.ByteInterval(address=0x1000, contents=b"!")
    tail = gtirb.DataBlock(offset=16, size=16)
    second = gtirb.ByteInterval(contents=b"x" * 32, blocks=[tail])

    with pytest.raises(PaddingError, match="incompatible alignment"):
        join_byte_intervals([first, second], alignment={second: 32, tail: 32})

    assert first.contents == b"!"
    assert tail.byte_interval is second
    assert second.contents == b"x" * 32


@pytest.mark.parametrize("boundary", [0, -1, 3])
def test_join_rejects_invalid_alignment(boundary):
    first = gtirb.ByteInterval(address=0x1000, contents=b"!")
    second = gtirb.ByteInterval(contents=b"x")
    with pytest.raises(PaddingError, match="power of two"):
        join_byte_intervals([first, second], alignment={second: boundary})
