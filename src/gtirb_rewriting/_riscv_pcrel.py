"""Preserve AUIPC instruction anchors across a rewriting transaction."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gtirb


@dataclass(frozen=True)
class _Pair:
    high: gtirb.SymAddrConst
    low: gtirb.SymAddrConst
    anchor: gtirb.Symbol
    high_bytes: bytes
    low_bytes: bytes


class RiscvPcrelPairs:
    # These private symbols are instruction anchors, never public entry labels.
    _PREFIX = ".L_gtirb_pcrel_"
    _A = gtirb.SymbolicExpression.Attribute

    def __init__(self, module: gtirb.Module):
        self.module = module
        self.pairs: List[_Pair] = []
        isa = module.isa.name
        info = module.aux_data.get("archInfo")
        if isa == "ValidButUnsupported" and info is not None:
            isa = info.data.get("ISA", isa)
        if str(isa).upper() not in ("RISCV32", "RISCV64"):
            return
        self.byte_order = (
            "big" if module.byte_order == gtirb.Module.ByteOrder.Big else "little"
        )

        for interval in module.byte_intervals:
            for offset, low in interval.symbolic_expressions.items():
                if not isinstance(low, gtirb.SymAddrConst) or not {
                    self._A.PCREL, self._A.LO
                }.issubset(low.attributes):
                    continue
                self.pairs.append(self._read_pair(interval, offset, low))

    def _read_pair(self, interval, offset, low) -> _Pair:
        anchor = low.symbol.referent
        if (
            low.offset
            or not isinstance(anchor, gtirb.CodeBlock)
            or anchor.byte_interval is None
        ):
            raise ValueError("RISC-V PC-relative LO has no instruction anchor")
        high_offset = anchor.offset + (anchor.size if low.symbol.at_end else 0)
        high = anchor.byte_interval.symbolic_expressions.get(high_offset)
        high_bytes = self._instruction_bytes(anchor.byte_interval, high_offset)
        if not self._is_high(high, high_bytes):
            raise ValueError("RISC-V PC-relative LO has a stale AUIPC HI anchor")
        return _Pair(
            high, low, low.symbol, high_bytes,
            self._instruction_bytes(interval, offset)
        )

    def _instruction_bytes(self, interval: gtirb.ByteInterval, offset: int) -> bytes:
        data = bytes(interval.contents[offset:offset + 4])
        if len(data) < 2:
            raise ValueError("RISC-V PC-relative instruction is truncated")
        size = 4 if int.from_bytes(data[:2], self.byte_order) & 3 == 3 else 2
        if len(data) < size:
            raise ValueError("RISC-V PC-relative instruction is truncated")
        return data[:size]

    def _is_high(self, expression, data: bytes) -> bool:
        if not isinstance(expression, gtirb.SymAddrConst) or len(data) != 4:
            return False
        word = int.from_bytes(data, self.byte_order)
        return word & 0x7F == 0x17 and (
            {self._A.PCREL, self._A.HI}.issubset(expression.attributes)
            or self._A.GOT in expression.attributes
            or self._A.TLSGD in expression.attributes
        )

    def restore(self) -> None:
        if not self.pairs:
            return
        # Interval edits move the existing expression objects. Track both ends
        # by identity, not equality: a newly inserted identical AUIPC is not
        # the original instruction. Output uses ordinary symbols, so no Python
        # identity or extra auxdata needs to survive serialization.
        tracked = {
            id(expression)
            for pair in self.pairs
            for expression in (pair.high, pair.low)
        }
        original_anchors = {pair.anchor: pair for pair in self.pairs}
        untracked_lows = []
        positions: Dict[int, List[Tuple[gtirb.ByteInterval, int]]] = {}
        for interval in self.module.byte_intervals:
            for offset, expression in interval.symbolic_expressions.items():
                if id(expression) in tracked:
                    positions.setdefault(id(expression), []).append((interval, offset))
                elif isinstance(expression, gtirb.SymAddrConst) and {
                    self._A.PCREL, self._A.LO
                }.issubset(expression.attributes):
                    untracked_lows.append((interval, offset, expression))

        anchors: Dict[int, gtirb.Symbol] = {}
        for pair in self.pairs:
            lows = positions.get(id(pair.low), [])
            if not lows:
                continue
            highs = positions.get(id(pair.high), [])
            if not highs:
                raise ValueError("RISC-V PC-relative HI was removed while its LO survives")
            if len(highs) != 1 or len(lows) != 1:
                raise ValueError("RISC-V PC-relative expression identity is ambiguous")
            high_interval, high_offset = highs[0]
            low_interval, low_offset = lows[0]
            if (
                self._instruction_bytes(high_interval, high_offset) != pair.high_bytes
                or self._instruction_bytes(low_interval, low_offset) != pair.low_bytes
                or not self._is_high(pair.high, pair.high_bytes)
            ):
                raise ValueError("RISC-V PC-relative instruction changed without replacing its relocation")

            anchor = pair.low.symbol
            block = anchor.referent
            if (
                anchor.name.startswith(self._PREFIX)
                and isinstance(block, gtirb.CodeBlock)
                and block.byte_interval is high_interval
                and block.offset + (block.size if anchor.at_end else 0) == high_offset
            ):
                continue
            if id(pair.high) in anchors:
                anchor = anchors[id(pair.high)]
            else:
                if not anchor.name.startswith(self._PREFIX):
                    anchor = gtirb.Symbol(name="", module=self.module)
                    anchor.name = self._PREFIX + anchor.uuid.hex
                if (
                    isinstance(block, gtirb.CodeBlock)
                    and block.size == 0
                    and set(block.references) == {anchor}
                    and not any(block.incoming_edges)
                    and not any(block.outgoing_edges)
                ):
                    block.byte_interval = high_interval
                    block.offset = high_offset
                else:
                    anchor.referent = gtirb.CodeBlock(
                        size=0, offset=high_offset, byte_interval=high_interval
                    )
                anchor.at_end = False
                anchors[id(pair.high)] = anchor
            low_interval.symbolic_expressions[low_offset] = gtirb.SymAddrConst(
                0, anchor, pair.low.attributes
            )

        # A replacement LO object cannot disappear from validation merely
        # because its identity is new. Reusing an original anchor still names
        # that anchor's original producer; changing the producer requires an
        # explicit new instruction anchor, not a search for a nearby AUIPC.
        for interval, offset, low in untracked_lows:
            pair = self._read_pair(interval, offset, low)
            original = original_anchors.get(pair.anchor)
            if original is not None and (
                pair.high is not original.high
                or pair.high_bytes != original.high_bytes
                or len(positions.get(id(original.high), [])) != 1
            ):
                raise ValueError(
                    "RISC-V PC-relative LO was replaced without a new instruction anchor"
                )
