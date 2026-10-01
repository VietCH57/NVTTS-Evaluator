import pytest

from nvtts_eval.data import NVParseError, NVParser


@pytest.fixture
def parser():
    return NVParser()


def ev(p):
    return [(e.type, e.gap_index) for e in p.events]


def test_basic_middle(parser):
    p = parser.parse("Hôm nay [laughter] tôi rất vui.")
    assert p.clean_text == "Hôm nay tôi rất vui."
    assert p.words == ("Hôm", "nay", "tôi", "rất", "vui.")
    assert ev(p) == [("laughter", 2)]


def test_start_and_end(parser):
    p = parser.parse("[throatclearing] xin chào các bạn [breathing]")
    assert ev(p) == [("throatclearing", 0), ("breathing", 4)]
    assert p.n_words == 4


def test_multiple_events_and_order(parser):
    p = parser.parse("a [breathing] b c [sniff] d")
    assert ev(p) == [("breathing", 1), ("sniff", 3)]


def test_adjacent_tags_share_a_gap_and_keep_order(parser):
    p = parser.parse("a [sniff] [breathing] b")
    assert ev(p) == [("sniff", 1), ("breathing", 1)]
    p2 = parser.parse("a [breathing] [sniff] b")
    assert ev(p2) == [("breathing", 1), ("sniff", 1)]


def test_tags_only_and_empty(parser):
    assert ev(parser.parse("[laughter]")) == [("laughter", 0)]
    assert parser.parse("[laughter]").clean_text == ""
    e = parser.parse("")
    assert e.clean_text == "" and e.events == () and e.n_words == 0


def test_whitespace_is_normalised(parser):
    p = parser.parse("  a   [laughter]\t b  ")
    assert p.clean_text == "a b" and ev(p) == [("laughter", 1)]


def test_tag_is_a_token_boundary_even_when_glued(parser):
    p = parser.parse("a[laughter]b")
    assert p.words == ("a", "b") and ev(p) == [("laughter", 1)]


def test_case_insensitive_tag_name(parser):
    assert ev(parser.parse("a [Laughter ] b")) == [("laughter", 1)]


def test_text_is_not_otherwise_normalised(parser):
    p = parser.parse("Chứ, [breathing] Đã xong")
    assert p.clean_text == "Chứ, Đã xong"      # case and punctuation untouched


def test_unknown_tag_policies():
    with pytest.raises(NVParseError):
        NVParser(unknown_policy="error").parse("a [cough] b")
    d = NVParser(unknown_policy="drop").parse("a [cough] b [sniff]")
    assert d.clean_text == "a b" and ev(d) == [("sniff", 2)] and d.unknown_tags == ("[cough]",)
    k = NVParser(unknown_policy="keep").parse("a [cough] b")
    assert ev(k) == [("cough", 1)]


@pytest.mark.parametrize("bad", ["a [laughter b", "a laughter] b", "a [[laughter]] b", "[a [sniff] b]"])
def test_malformed_brackets_raise(parser, bad):
    with pytest.raises(NVParseError):
        parser.parse(bad)


def test_empty_tag_is_unknown(parser):
    with pytest.raises(NVParseError):
        parser.parse("a [] b")


def test_roundtrip_through_tagged_text(parser):
    raw = "[throatclearing] a b  [sniff]   [breathing] c d [laughter]"
    p = parser.parse(raw)
    again = parser.parse(p.tagged_text())
    assert again.events == p.events and again.clean_text == p.clean_text
    assert p.tagged_text() == "[throatclearing] a b [sniff] [breathing] c d [laughter]"


def test_real_sentence_from_data(parser):
    p = parser.parse("mà anh ta cùng với man city bảo vệ thành công chức vô địch champions league [breathing] thì rất có thể là anh ta")
    assert ev(p) == [("breathing", 16)]   # counted by hand: 16 syllables precede the tag


def test_type_error(parser):
    with pytest.raises(TypeError):
        parser.parse(None)
