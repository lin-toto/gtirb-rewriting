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
import dataclasses
import itertools
import logging
import time
from typing import Iterable, List, Mapping, MutableMapping, Optional

import gtirb

import gtirb_rewriting._auxdata as _auxdata

from ._adt import OffsetMapping
from ._auxdata_offsetmap import OFFSETMAP_AUX_DATA_TABLES
from .abi import ABI
from .utils import align_address

logger = logging.getLogger("gtirb_rewriting")

_LARGE_INTERVAL_COUNT = 10_000
_PROGRESS_INTERVAL = 100_000


class PaddingError(Exception):
    """Indicates an error inserting padding to reach a desired alignment."""


@dataclasses.dataclass
class BlockGroup:
    """A group of overlapping blocks."""

    begin: int
    """Offset of the first block in the group."""
    end: int
    """First offset past the end of the last block in the group."""
    blocks: List[gtirb.ByteBlock]
    """Collection of blocks in the group."""


def split_byte_interval(
    interval: gtirb.ByteInterval,
    alignment: Optional[MutableMapping[gtirb.Node, int]] = None,
    tables: Optional[Iterable[OffsetMapping[object]]] = None,
    isolated_blocks: Optional[Iterable[gtirb.ByteBlock]] = None,
) -> List[gtirb.ByteInterval]:
    """Split a ByteInterval into groups of blocks.

    By default, the original interval will hold the first block (ordered by
    offset), and each remaining block will be in a new byte interval. When
    ``isolated_blocks`` is provided, only those blocks are isolated;
    consecutive unselected blocks remain together. Any bytes outside of a
    block will be included in the interval containing the preceding block; the
    first interval will contain the bytes before and after the first block.

    Because overlapping blocks share the bytes where they overlap, some
    intervals may contain more than one block after the split. These intervals
    will contain the smallest number of blocks possible without duplicating
    bytes.

    :param interval:  byte interval to split
    :param alignment:  optional table of alignments for blocks and intervals
    :param tables:  optional collection of offset mappings to update
    :param isolated_blocks:  optional blocks whose overlapping block groups
        should be placed in separate intervals; other consecutive groups are
        kept together unless an alignment boundary requires a split
    :returns:  list of byte intervals containing the blocks in the original
        interval
    """
    started = time.perf_counter()
    block_count = len(interval.blocks)
    is_large = block_count >= _LARGE_INTERVAL_COUNT
    section_name = interval.section.name if interval.section else "none"
    if is_large:
        logger.info(
            "split: begin section=%s blocks=%d size=%d selected=%s",
            section_name,
            block_count,
            interval.size,
            "all" if isolated_blocks is None else "scoped",
        )

    if tables is None:
        tables = []
        module = interval.module
        if module is not None:
            for table_def in OFFSETMAP_AUX_DATA_TABLES:
                table = table_def.get(module)
                if table:
                    tables.append(table)

    # Group overlapping blocks so they can be processed as a unit.
    groups: List[BlockGroup] = []
    sorted_blocks = sorted(interval.blocks, key=lambda b: b.offset)
    if is_large:
        logger.info(
            "split: sorted %d blocks in %.1fs",
            block_count,
            time.perf_counter() - started,
        )
    for block_idx, block in enumerate(sorted_blocks, 1):
        block_end = block.offset + block.size
        if groups == [] or groups[-1].end <= block.offset:
            groups.append(BlockGroup(block.offset, block_end, [block]))
        else:
            groups[-1].end = max(groups[-1].end, block_end)
            groups[-1].blocks.append(block)
        if is_large and block_idx % _PROGRESS_INTERVAL == 0:
            logger.info(
                "split: grouped %d/%d blocks into %d groups in %.1fs",
                block_idx,
                block_count,
                len(groups),
                time.perf_counter() - started,
            )

    if isolated_blocks is not None:
        isolated_block_set = set(isolated_blocks)
        if any(
            block.byte_interval is not interval
            for block in isolated_block_set
        ):
            raise ValueError("block is not part of the byte interval")

        coalesced_groups: List[BlockGroup] = []
        previous_is_isolated = False
        for group in groups:
            group_is_isolated = any(
                block in isolated_block_set for block in group.blocks
            )
            group_starts_aligned = alignment is not None and any(
                block in alignment for block in group.blocks
            )
            if (
                coalesced_groups
                and not previous_is_isolated
                and not group_is_isolated
                and not group_starts_aligned
            ):
                coalesced_groups[-1].end = group.end
                coalesced_groups[-1].blocks.extend(group.blocks)
            else:
                coalesced_groups.append(group)
            previous_is_isolated = group_is_isolated
        groups = coalesced_groups
        if is_large:
            logger.info(
                "split: coalesced to %d groups in %.1fs",
                len(groups),
                time.perf_counter() - started,
            )

    # Process groups in decreasing offset order, but skip the first group
    # because it will stay in the original interval.
    if groups:
        groups.reverse()
        groups.pop()

    # Create the new byte interval for each group of blocks.
    intervals: List[gtirb.ByteInterval] = []
    original_contents = interval.contents
    original_size = interval.size
    offset = original_size
    group_count = len(groups)
    for group_idx, group in enumerate(groups, 1):
        new_interval = gtirb.ByteInterval(
            contents=original_contents[group.begin : offset],
            size=max(offset - group.begin, 0),
        )
        new_interval.section = interval.section

        new_interval.address = group.blocks[0].address
        for block in group.blocks:
            block.offset -= group.begin
            block.byte_interval = new_interval
        intervals.append(new_interval)

        offset = min(original_size, group.begin)
        if is_large and group_idx % _PROGRESS_INTERVAL == 0:
            logger.info(
                "split: created %d/%d intervals in %.1fs",
                group_idx,
                group_count,
                time.perf_counter() - started,
            )
    interval.initialized_size = min(len(original_contents), offset)
    interval.size = min(original_size, offset)
    intervals.append(interval)

    # Transfer symbolic expressions and table items to the new intervals.
    symexprs = OffsetMapping()
    symexprs[interval] = dict(interval.symbolic_expressions)
    for table_idx, table in enumerate(
        itertools.chain((symexprs,), tables), 1
    ):
        if is_large:
            logger.info(
                "split: transferring offset table %d in %.1fs",
                table_idx,
                time.perf_counter() - started,
            )
        items = sorted(table.get(interval, {}).items())
        for group, new_interval in zip(groups, intervals):
            while items != [] and items[-1][0] >= group.begin:
                off, value = items.pop()
                del table[interval][off]
                if new_interval not in table:
                    table[new_interval] = {}
                table[new_interval].update({off - group.begin: value})
    for new_interval in intervals:
        if new_interval in symexprs:
            new_interval.symbolic_expressions = symexprs[new_interval]

    intervals.reverse()
    if is_large:
        logger.info(
            "split: complete section=%s intervals=%d in %.1fs",
            section_name,
            len(intervals),
            time.perf_counter() - started,
        )
    return intervals


def join_byte_intervals(
    intervals: List[gtirb.ByteInterval],
    nop: Optional[bytes] = None,
    alignment: Optional[Mapping[gtirb.Node, int]] = None,
    tables: Optional[Iterable[OffsetMapping[object]]] = None,
    nop_encodings: Optional[Mapping[gtirb.CodeBlock.DecodeMode, bytes]] = None,
) -> gtirb.ByteInterval:
    """Concatenate a list of byte intervals.

    The first interval in the given list will be trated as the destination. The
    contents (bytes and byte_blocks) of all other intervals will be
    concatenated onto the end of the destination interval in the order they
    appear in the list.

    Padding will be inserted between subsequent intervals so that the address
    of the first block of each interval (or of the interval itself if it
    contains no blocks) is properly aligned. Addresses are calculated based on
    the address of the destination block, or 0 if it has no address. If the
    alignment mapping is not specified, the "alignment" aux data for each
    interval's module, if any, will be used.

    The symbolic expressions will be transfered to the destination module,
    adjusted to retain their positions relative to their original byte
    interval. In addition, any tables given will be updated by relocating
    Offsets into each concatenated interval to refer to the corresponding
    Offset into the destination interval. If no tables are provided, a default
    set of aux data will be updated; pass an empty sequence of tables to
    prevent any tables from being updated.

    NB: This function destructively removes the blocks, bytes, and symbolic
    expressions from the other intervals when they are added to the
    destination, but it does not remove the intervals from their sections.

    :param intervals:  list of byte intervals to concatenate
    :param nop:  bytes representing a single nop instruction in the default
        decode mode
    :param alignment:  table of alignments for blocks and intervals
    :param tables:  collection of offset mappings to update
    :param nop_encodings:  nop bytes to use in different decode; if the mapping
        specifies bytes for the default decode mode, they will supercede the
        bytes in the `nop` argument
    """
    interval_count = len(intervals)
    if interval_count < 2:
        return intervals[0]

    started = time.perf_counter()
    is_large = interval_count >= _LARGE_INTERVAL_COUNT

    nop_encodings = dict(nop_encodings) if nop_encodings else {}
    if nop is not None:
        nop_encodings.setdefault(gtirb.CodeBlock.DecodeMode.Default, nop)

    destination = intervals[0]
    section_name = (
        destination.section.name if destination.section else "none"
    )
    if is_large:
        logger.info(
            "join: begin section=%s intervals=%d",
            section_name,
            interval_count,
        )

    source_table_entries = []
    if tables is None:
        # This is a bit hacky, but to avoid assuming that the byte intervals
        # are all in the same module, the tables are a list of dictionaries
        # that map intervals to (displacement to value) dicts. Each interval
        # added at this stage will map to the mutable mapping returned by
        # indexing an OffsetMapping, which means the original aux data will be
        # updated when modifying that sub-dict.
        tables = []
        for table_def in OFFSETMAP_AUX_DATA_TABLES:
            if is_large:
                logger.info(
                    "join: scanning offset table %s across %d intervals "
                    "in %.1fs",
                    table_def.name,
                    interval_count,
                    time.perf_counter() - started,
                )
            table = {}
            for bi in intervals:
                if bi.module is not None:
                    aux_data = table_def.get(bi.module)
                    if aux_data is not None and bi in aux_data:
                        table[bi] = aux_data[bi]
                        if bi is not destination:
                            source_table_entries.append((aux_data, bi))
            if len(table) > 0:
                if destination not in table:
                    assert destination.module is not None
                    destination_data = table_def.get_or_insert(
                        destination.module
                    )
                    destination_data[destination] = {}
                    table[destination] = destination_data[destination]
                tables.append(table)  # type: ignore # per above this is hacky

    intervals = intervals[1:]

    address = 0
    if destination.address is not None:
        address = destination.address
    address += destination.size
    last_block = max(destination.blocks, key=lambda b: b.offset, default=None)
    last_module = last_block.module if last_block is not None else None
    contents = bytearray(destination.contents)

    def insert_padding(size):
        if size == 0:
            return
        if isinstance(last_block, gtirb.CodeBlock):
            if last_block.decode_mode in nop_encodings:
                pad_bytes = nop_encodings[last_block.decode_mode]
            elif last_module is not None:
                # TODO: get NOP encoding for the current decode_mode here.
                pad_bytes = ABI.get(last_module).nop()
            else:
                raise PaddingError("cannot determine nop instruction")
            size, remainder = divmod(size, len(pad_bytes))
            if remainder != 0:
                raise PaddingError("nop does not fit evenly in padding")
        else:
            pad_bytes = b"\x00"

        contents.extend(pad_bytes * size)
        # The pretty-printer won't print the padding bytes unless they
        # are contained in blocks, add a block covering anything not
        # yet covered by the last block.
        if last_block is not None:
            padding_block_offset = last_block.offset + last_block.size
            padding_block_size = (
                len(contents) - padding_block_offset
            )
        else:
            padding_block_offset = 0
            padding_block_size = len(contents)
        if padding_block_size > 0:
            if isinstance(last_block, gtirb.CodeBlock):
                padding = gtirb.CodeBlock(
                    offset=padding_block_offset,
                    size=padding_block_size,
                    decode_mode=last_block.decode_mode,
                )
            else:
                padding = gtirb.DataBlock(
                    offset=padding_block_offset, size=padding_block_size
                )
            padding.byte_interval = destination

    symexprs = OffsetMapping()
    deltas = {}
    source_interval_count = len(intervals)
    for interval_idx, interval in enumerate(intervals, 1):
        # Fill in any uninitialized bytes before appending.
        insert_padding(destination.size - len(contents))

        # Align the first block if possible, or the interval if not.
        if alignment is not None:
            module_alignment = alignment
        elif interval.module is not None and _auxdata.alignment.exists(
            interval.module
        ):
            module_alignment = _auxdata.alignment.get_or_insert(
                interval.module
            )
        else:
            module_alignment = {}
        node = min(
            (b for b in interval.blocks if b in module_alignment),
            key=lambda b: b.offset,
            default=interval,
        )
        if node == interval:
            offset = 0
        else:
            assert isinstance(node, gtirb.ByteBlock)
            offset = node.offset
        boundary = module_alignment.get(node, 1)
        size = align_address(address + offset, boundary) - (address + offset)
        insert_padding(size)
        address += size
        destination.size += size

        # Cache the delta for updating the symbolic expression offsets and the
        # new last block in case we need more padding.
        deltas[interval] = len(contents)
        if interval.symbolic_expressions:
            symexprs[interval] = dict(interval.symbolic_expressions)
        last_block = max(
            interval.blocks, default=last_block, key=lambda b: b.offset
        )
        if last_block is not None and last_block.module is not None:
            last_module = last_block.module

        # Transfer the bytes and blocks to the new intervals.
        address += interval.size
        destination.size += interval.size
        contents.extend(interval.contents)
        for block in tuple(interval.blocks):
            block.offset += deltas[interval]
            block.byte_interval = destination

        interval.initialized_size = 0
        interval.symbolic_expressions.clear()
        if is_large and interval_idx % _PROGRESS_INTERVAL == 0:
            logger.info(
                "join: appended %d/%d intervals bytes=%d in %.1fs",
                interval_idx,
                source_interval_count,
                len(contents),
                time.perf_counter() - started,
            )

    if is_large:
        logger.info(
            "join: materializing %d content bytes in %.1fs",
            len(contents),
            time.perf_counter() - started,
        )
    destination.contents = contents
    destination.initialized_size = len(contents)
    if is_large:
        logger.info(
            "join: content materialized in %.1fs",
            time.perf_counter() - started,
        )

    # Update offsets to refer to the destination interval.
    for table_idx, table in enumerate(
        itertools.chain((symexprs,), tables), 1
    ):
        if is_large:
            logger.info(
                "join: updating offset table %d in %.1fs",
                table_idx,
                time.perf_counter() - started,
            )
        destination_items = table.get(destination)
        if destination_items is None:
            destination_items = {}
            table[destination] = destination_items
        for interval_idx, interval in enumerate(intervals, 1):
            old = table.get(interval)
            if old:
                destination_items.update(
                    (k + deltas[interval], v) for k, v in old.items()
                )
            if interval in table:
                del table[interval]
            if is_large and interval_idx % _PROGRESS_INTERVAL == 0:
                logger.info(
                    "join: offset table %d processed %d/%d intervals "
                    "in %.1fs",
                    table_idx,
                    interval_idx,
                    source_interval_count,
                    time.perf_counter() - started,
                )
    for table, interval in source_table_entries:
        if interval in table:
            del table[interval]
    destination.symbolic_expressions.update(symexprs[destination])

    if is_large:
        logger.info(
            "join: complete section=%s intervals=%d bytes=%d in %.1fs",
            section_name,
            interval_count,
            len(contents),
            time.perf_counter() - started,
        )

    return destination
