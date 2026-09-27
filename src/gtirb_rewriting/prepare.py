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
import contextlib
import logging
import time
from typing import Dict, Iterable, Iterator, Optional, Set

import gtirb
from gtirb_layout import (
    assign_integral_symbols,
    is_module_layout_required,
    layout_module,
)

import gtirb_rewriting._auxdata as _auxdata

from ._riscv_pcrel import RiscvPcrelPairs
from .intervalutils import (
    _alignment_requirement,
    join_byte_intervals,
    split_byte_interval,
)

logger = logging.getLogger("gtirb_rewriting")

_INTERVAL_PROGRESS_INTERVAL = 100
_SHN_ABS = 0xFFF1


@contextlib.contextmanager
def _preserve_absolute_symbols(module: gtirb.Module) -> Iterator[None]:
    # gtirb-layout 1.x does not distinguish SHN_ABS values from addresses.
    # Hide only their numeric payload while that dependency assigns referents;
    # patch generation must continue to see the original values.
    info = _auxdata.elf_symbol_info.get(module) or {}
    values = {
        symbol: symbol.value
        for symbol, attributes in info.items()
        if symbol.module is module
        and symbol.value is not None
        and attributes[4] == _SHN_ABS
    }
    for symbol in values:
        symbol.value = None
    try:
        yield
    finally:
        for symbol, value in values.items():
            symbol.value = value


def _assign_integral_symbols(module: gtirb.Module) -> None:
    with _preserve_absolute_symbols(module):
        assign_integral_symbols(module)
    if module.file_format != gtirb.Module.FileFormat.ELF:
        return
    info = _auxdata.elf_symbol_info.get_or_insert(module)
    for symbol in module.symbols:
        if symbol.value is not None:
            attributes = info.get(symbol, (0, "NOTYPE", "LOCAL", "DEFAULT", 0))
            # An inferred local value without section provenance must not bind
            # to an unrelated interval after layout. Keep real section indices
            # and undefined external bindings intact, even outside byte extents.
            if attributes[4] == 0 and attributes[2] == "LOCAL":
                info[symbol] = (*attributes[:4], _SHN_ABS)


def _layout_module(module: gtirb.Module) -> None:
    """Keep every explicit alignment when the layout dependency moves runs.

    gtirb-layout 1.x uses only the first aligned block in each interval. Give
    it the strongest compatible constraint while laying out, then restore all
    original metadata. This preserves both the interval order/defaults chosen
    by the dependency and the relative offsets of immutable data runs.
    """
    alignment = _auxdata.alignment.get(module)
    if not alignment:
        with _preserve_absolute_symbols(module):
            layout_module(module)
        return

    # Validate before creating temporary nodes or changing the aux data.
    anchors = []
    for interval in module.byte_intervals:
        modulus, _ = _alignment_requirement(interval, alignment)
        aligned_blocks = [
            block for block in interval.blocks if block in alignment
        ]
        if aligned_blocks:
            block = max(aligned_blocks, key=alignment.__getitem__)
            if alignment[block] == modulus:
                anchors.append((interval, block, modulus))
                continue
        if interval in alignment:
            anchors.append((interval, None, modulus))

    auxiliary = module.aux_data[_auxdata.alignment.name]
    reduced_alignment = {}
    temporary_blocks = []
    try:
        for interval, block, modulus in anchors:
            if block is None:
                # The dependency ignores ByteInterval alignment entries. A
                # temporary zero-sized anchor expresses their start residue
                # without introducing bytes or any persistent block/symbol.
                block = gtirb.DataBlock(byte_interval=interval)
                temporary_blocks.append(block)
            reduced_alignment[block] = modulus
        auxiliary.data = reduced_alignment
        with _preserve_absolute_symbols(module):
            layout_module(module)
    finally:
        auxiliary.data = alignment
        for block in temporary_blocks:
            block.byte_interval = None


def _should_log_interval(index: int, total: int) -> bool:
    return (
        total <= 20
        or index == 1
        or index == total
        or index % _INTERVAL_PROGRESS_INTERVAL == 0
    )


@contextlib.contextmanager
def prepare_for_rewriting(
    module: gtirb.Module,
    nop: bytes,
    blocks: Optional[Iterable[gtirb.ByteBlock]] = None,
) -> Iterator[None]:
    """Pre-compute rewrite data for selected blocks, or all if None.

    :param module: module that will be rewritten
    :param nop: default-decode-mode nop encoding
    :param blocks: blocks that may be rewritten; overlapping groups containing
        these blocks are isolated while unrelated block runs remain together
    """
    started = time.perf_counter()
    selected_blocks = None if blocks is None else set(blocks)
    logger.info(
        "prepare: begin mode=%s selected_blocks=%s",
        "all" if selected_blocks is None else "scoped",
        "all" if selected_blocks is None else len(selected_blocks),
    )

    phase_started = time.perf_counter()
    logger.info("prepare: assigning integral symbols")
    _assign_integral_symbols(module)
    pcrel_pairs = RiscvPcrelPairs(module)
    if is_module_layout_required(module):
        logger.info("prepare: laying out input module")
        _layout_module(module)
    logger.info(
        "prepare: input layout ready in %.1fs",
        time.perf_counter() - phase_started,
    )

    alignment = (
        {} if module.file_format == gtirb.Module.FileFormat.ELF else None
    )
    if _auxdata.alignment.exists(module):
        alignment = _auxdata.alignment.get_or_insert(module)

    partitions = []
    phase_started = time.perf_counter()
    if selected_blocks is None:
        source_intervals = tuple(module.byte_intervals)
        logger.info(
            "prepare: splitting all %d byte intervals",
            len(source_intervals),
        )
        for interval_idx, interval in enumerate(source_intervals, 1):
            if _should_log_interval(interval_idx, len(source_intervals)):
                logger.info(
                    "prepare: splitting interval %d/%d section=%s "
                    "blocks=%d",
                    interval_idx,
                    len(source_intervals),
                    interval.section.name if interval.section else "none",
                    len(interval.blocks),
                )
            partitions.append(split_byte_interval(interval, alignment))
    else:
        blocks_by_interval: Dict[
            gtirb.ByteInterval, Set[gtirb.ByteBlock]
        ] = {}
        for block in selected_blocks:
            interval = block.byte_interval
            if interval is None or interval.module is not module:
                raise ValueError("block is not part of the module")
            blocks_by_interval.setdefault(interval, set()).add(block)
        logger.info(
            "prepare: splitting %d byte intervals containing %d selected "
            "blocks",
            len(blocks_by_interval),
            len(selected_blocks),
        )
        for interval_idx, (interval, isolated_blocks) in enumerate(
            blocks_by_interval.items(), 1
        ):
            if _should_log_interval(interval_idx, len(blocks_by_interval)):
                logger.info(
                    "prepare: splitting interval %d/%d section=%s "
                    "blocks=%d selected=%d",
                    interval_idx,
                    len(blocks_by_interval),
                    interval.section.name if interval.section else "none",
                    len(interval.blocks),
                    len(isolated_blocks),
                )
            partitions.append(
                split_byte_interval(
                    interval,
                    alignment,
                    isolated_blocks=isolated_blocks,
                )
            )
    logger.info(
        "prepare: split complete partitions=%d intervals=%d in %.1fs",
        len(partitions),
        sum(len(partition) for partition in partitions),
        time.perf_counter() - phase_started,
    )

    yield

    phase_started = time.perf_counter()
    logger.info("prepare: rejoining %d partitions", len(partitions))
    for partition_idx, partition in enumerate(partitions, 1):
        if _should_log_interval(partition_idx, len(partitions)):
            logger.info(
                "prepare: rejoining partition %d/%d intervals=%d",
                partition_idx,
                len(partitions),
                len(partition),
            )
        join_byte_intervals(partition, nop, alignment)
        for interval in partition[1:]:
            interval.section = None
    logger.info(
        "prepare: rejoin complete in %.1fs",
        time.perf_counter() - phase_started,
    )

    pcrel_pairs.restore()
    if is_module_layout_required(module):
        phase_started = time.perf_counter()
        logger.info("prepare: laying out rewritten module")
        _assign_integral_symbols(module)
        _layout_module(module)
        logger.info(
            "prepare: rewritten module layout complete in %.1fs",
            time.perf_counter() - phase_started,
        )

    logger.info("prepare: complete in %.1fs", time.perf_counter() - started)
