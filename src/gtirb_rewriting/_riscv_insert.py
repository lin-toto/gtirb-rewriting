"""Keep insertions out of indivisible RISC-V relocation/call pairs."""

from bisect import bisect_left

import gtirb

from .decoder import module_is_riscv64

_A = gtirb.SymbolicExpression.Attribute


def _protected_hi(symbolic):
    return isinstance(symbolic, gtirb.SymAddrConst) and (
        {_A.HI, _A.PCREL}.issubset(symbolic.attributes)
        or _A.GOT in symbolic.attributes
        or _A.TLSGD in symbolic.attributes
    )


def _call_relocation(symbolic):
    if not isinstance(symbolic, gtirb.SymAddrConst):
        return False
    # The instruction check below also requires an AUIPC/JALR pair. Lifted
    # calls can have separate PCREL HI/LO instead of a CALL/PLT expression.
    return (
        _A.PLT in symbolic.attributes
        or {_A.HI, _A.PCREL}.issubset(symbolic.attributes)
        or not symbolic.attributes.intersection(
            {_A.HI, _A.LO, _A.GOT, _A.PCREL, _A.TLSGD}
        )
    )


def _auipc_jalr_pair(interval, offset):
    contents = interval.contents
    if offset < 0 or offset + 8 > len(contents):
        return False
    first = int.from_bytes(contents[offset : offset + 4], "little")
    second = int.from_bytes(contents[offset + 4 : offset + 8], "little")
    if (first & 0x7F) != 0x17 or (second & 0x7F) != 0x67:
        return False
    register = (first >> 7) & 0x1F
    return register != 0 and register == (second >> 15) & 0x1F


class RiscvInsertionLocations:
    """Indexes valid for one RewritingContext's registration phase."""

    def __init__(self):
        self._relocation_offsets = {}
        self._blocks_ending_at = {}

    def clear(self):
        self._relocation_offsets.clear()
        self._blocks_ending_at.clear()

    def _offsets(self, block):
        interval = block.byte_interval
        offsets = self._relocation_offsets.get(interval)
        if offsets is None:
            offsets = tuple(sorted(interval.symbolic_expressions))
            self._relocation_offsets[interval] = offsets
        start = bisect_left(offsets, block.offset)
        end = bisect_left(offsets, block.offset + block.size, start)
        return offsets[start:end]

    def _predecessors(self, interval, offset):
        index = self._blocks_ending_at.get(interval)
        if index is None:
            index = {}
            for block in interval.blocks:
                if isinstance(block, gtirb.CodeBlock) and block.size:
                    index.setdefault(block.offset + block.size, []).append(
                        block
                    )
            for blocks in index.values():
                blocks.sort(
                    key=lambda b: (b.offset, str(b.uuid)), reverse=True
                )
            self._blocks_ending_at[interval] = index
        return index.get(offset, ())

    def _safe_offset(self, block, offset):
        interval = block.byte_interval
        position = block.offset + offset
        expressions = interval.symbolic_expressions
        offsets = self._offsets(block)
        for high in offsets:
            if (
                _call_relocation(expressions[high])
                and _auipc_jalr_pair(interval, high)
                and high < position < high + 8
            ):
                return high - block.offset

        for index, high in enumerate(offsets):
            if not _protected_hi(expressions[high]):
                continue
            low = None
            for candidate in offsets[index + 1 :]:
                expr = expressions[candidate]
                if _protected_hi(expr):
                    break
                # An absolute LO may belong to already inserted coverage
                # instrumentation, not to this PC-relative producer.
                if (
                    isinstance(expr, gtirb.SymAddrConst)
                    and {_A.LO, _A.PCREL}.issubset(expr.attributes)
                ):
                    low = candidate
                    break
            if low is None or not high < position <= low:
                continue
            word = int.from_bytes(interval.contents[low : low + 4], "little")
            if word & 0x7F in (0x03, 0x23, 0x07, 0x27):
                # Memory accesses have side effects: a pre-store log must
                # remain before the store. Their explicit pinned PCREL anchor
                # already makes insertion between HI and LO safe.
                return offset
            size = 2 if interval.contents[low] & 3 != 3 else 4
            return min(low + size, block.offset + block.size) - block.offset
        return offset

    def resolve(self, block: gtirb.ByteBlock, offset: int):
        if (
            not isinstance(block, gtirb.CodeBlock)
            or not module_is_riscv64(block.module)
            or block.byte_interval is None
            or not 0 <= offset <= block.size
        ):
            return block, offset
        if offset == 0 and block.offset >= 4:
            # Older lifts may split a direct call at JALR. Until their producer
            # keeps it intact, the insertion still belongs before AUIPC.
            interval = block.byte_interval
            high = block.offset - 4
            if (
                _call_relocation(interval.symbolic_expressions.get(high))
                and _auipc_jalr_pair(interval, high)
            ):
                for prefix in self._predecessors(interval, block.offset):
                    if prefix.offset <= high < prefix.offset + prefix.size:
                        return prefix, self._safe_offset(
                            prefix, high - prefix.offset
                        )
        return block, self._safe_offset(block, offset)
