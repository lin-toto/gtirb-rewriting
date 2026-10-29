"""RISC-V assembly and CFG generation require no application monkeypatch."""

import gtirb
import pytest
from gtirb_test_helpers import add_proxy_block, add_symbol, create_test_module

from gtirb_rewriting import Assembler, UndefSymbolError

A = gtirb.SymbolicExpression.Attribute


def make_assembler(**kwargs):
    _, module = create_test_module(
        gtirb.Module.FileFormat.ELF, gtirb.Module.ISA.ValidButUnsupported
    )
    module.aux_data["archInfo"] = gtirb.AuxData(
        {"ISA": "RISCV64"}, "mapping<string,string>"
    )
    add_symbol(module, "target", add_proxy_block(module))
    return Assembler(module, **kwargs)


@pytest.mark.parametrize(
    "assembly, attributes, addend",
    [
        ("lui t0, %hi(target+24)", {A.HI}, 24),
        ("addi t0,t0,%lo(target-16)", {A.LO}, -16),
        ("sd t0,%lo(target)(t1)", {A.LO}, 0),
        ("auipc t0,%pcrel_hi(target)", {A.HI, A.PCREL}, 0),
        ("addi t0,t0,%pcrel_lo(target)", {A.LO, A.PCREL}, 0),
        ("sd t0,%pcrel_lo(target)(t1)", {A.LO, A.PCREL}, 0),
        ("auipc t0,%got_pcrel_hi(target)", {A.HI, A.PCREL, A.GOT}, 0),
        ("call target@plt", {A.PLT}, 0),
        ("tail target@plt", {A.PLT}, 0),
    ],
)
def test_target_relocations(assembly, attributes, addend):
    assembler = make_assembler()
    assembler.assemble(assembly)
    result = assembler.finalize()
    (expression,) = result.text_section.symbolic_expressions.values()
    assert expression.symbol.name == "target"
    assert expression.offset == addend
    assert expression.attributes == attributes


def test_plt_external_and_undefined_address():
    assembler = make_assembler()
    assembler.assemble("call external@plt")
    result = assembler.finalize()
    (expr,) = result.text_section.symbolic_expressions.values()
    assert expr.symbol.name == "external"
    assert expr.symbol.referent in result.proxies
    with pytest.raises(UndefSymbolError):
        make_assembler().assemble("lui t0,%hi(missing)")


def test_direct_jump_has_no_call_or_fallthrough_edge():
    assembler = make_assembler()
    assembler.assemble("j target\nnop")
    result = assembler.finalize()
    block = result.text_section.blocks[0]
    (edge,) = result.cfg.out_edges(block)
    assert edge.label == gtirb.Edge.Label(gtirb.Edge.Type.Branch)


def test_indirect_call_has_fallthrough():
    assembler = make_assembler()
    assembler.assemble("jalr ra,0(t0)\nnop")
    result = assembler.finalize()
    assert {edge.label.type for edge in result.cfg} == {
        gtirb.Edge.Type.Call,
        gtirb.Edge.Type.Fallthrough,
    }
    edge = next(
        edge for edge in result.cfg if edge.label.type == gtirb.Edge.Type.Call
    )
    assert not edge.label.direct


def test_reused_and_independent_assemblers_keep_their_source_operands():
    first = make_assembler(allow_undef_symbols=True)
    second = make_assembler(allow_undef_symbols=True)
    first.assemble("lui t0,%hi(first)")
    second.assemble("lui t0,%hi(second)")
    first.assemble("addi t0,t0,%lo(first)")
    first_result = first.finalize()
    second_result = second.finalize()
    assert [
        e.symbol.name
        for e in first_result.text_section.symbolic_expressions.values()
    ] == ["first", "first"]
    assert [
        e.symbol.name
        for e in second_result.text_section.symbolic_expressions.values()
    ] == ["second"]


@pytest.mark.parametrize("assembly, names, offsets", [
    ("call target@plt # %lo(decoy)", ["target"], [0]),
    ("lui t0,%hi(first); lui t1,%hi(second)",
     ["first", "second"], [0, 0]),
    ("lui t0,%hi(target + 24)", ["target"], [24]),
])
def test_source_recovery_uses_current_statement(assembly, names, offsets):
    assembler = make_assembler(allow_undef_symbols=True)
    assembler.assemble(assembly)
    section = assembler.finalize().text_section
    expressions = list(section.symbolic_expressions.values())
    assert [expr.symbol.name for expr in expressions] == names
    assert [expr.offset for expr in expressions] == offsets


@pytest.mark.parametrize("prefix", ["", "lui t1,%hi(other); "])
def test_opaque_macro_operand_is_rejected(prefix):
    from gtirb_rewriting import UnsupportedAssemblyError
    assembler = make_assembler(allow_undef_symbols=True)
    with pytest.raises(UnsupportedAssemblyError):
        assembler.assemble(
            prefix + ".macro load arg\n lui t0,%hi(\\arg)\n.endm\nload target"
        )


@pytest.mark.parametrize("assembly, edge_type", [
    ("jal zero,target", gtirb.Edge.Type.Branch),
    ("jal t0,target", gtirb.Edge.Type.Call),
    ("jalr zero,t1,0", gtirb.Edge.Type.Branch),
    ("jalr zero,t0,4", gtirb.Edge.Type.Branch),
    ("jr t1", gtirb.Edge.Type.Branch),
    ("jr t0", gtirb.Edge.Type.Return),
    ("ret", gtirb.Edge.Type.Return),
    (".option rvc\nc.j target", gtirb.Edge.Type.Branch),
    (".option rvc\nc.jr t1", gtirb.Edge.Type.Branch),
    (".option rvc\nc.jr ra", gtirb.Edge.Type.Return),
    (".option rvc\nc.jalr t0", gtirb.Edge.Type.Call),
])
def test_transfer_register_semantics(assembly, edge_type):
    assembler = make_assembler()
    assembler.assemble(assembly + "\nnop")
    result = assembler.finalize()
    expected = {edge_type}
    if edge_type == gtirb.Edge.Type.Call:
        expected.add(gtirb.Edge.Type.Fallthrough)
    assert {edge.label.type for edge in result.cfg} == expected
