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

from .intervalutils import join_byte_intervals, split_byte_interval

logger = logging.getLogger("gtirb_rewriting")

_INTERVAL_PROGRESS_INTERVAL = 100


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
    if is_module_layout_required(module):
        logger.info("prepare: laying out input module")
        layout_module(module)
    else:
        logger.info("prepare: assigning integral symbols")
        assign_integral_symbols(module)
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

    if is_module_layout_required(module):
        phase_started = time.perf_counter()
        logger.info("prepare: laying out rewritten module")
        layout_module(module)
        logger.info(
            "prepare: rewritten module layout complete in %.1fs",
            time.perf_counter() - phase_started,
        )

    logger.info("prepare: complete in %.1fs", time.perf_counter() - started)
