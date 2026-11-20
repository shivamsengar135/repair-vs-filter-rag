import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_cove", Path(__file__).resolve().parents[1] / "scripts" / "run_cove.py")
cove = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cove)


def test_parse_plan():
    text = ("Here are the facts:\n1. Marie Curie was born in Warsaw. ||| Where was Marie Curie born?\n"
            "- She won two Nobel Prizes ||| How many Nobel Prizes did Marie Curie win?\n"
            "a line without separator\n ||| orphan question?\nab ||| x\n")
    assert cove.parse_plan(text) == [("Marie Curie was born in Warsaw.", "Where was Marie Curie born?"),
                                     ("She won two Nobel Prizes", "How many Nobel Prizes did Marie Curie win?")]
    assert cove.parse_plan("") == []
    assert len(cove.parse_plan("\n".join(f"fact {i} ||| question {i}?" for i in range(100)), max_pairs=7)) == 7


def test_parse_crosscheck():
    f = "Marie Curie was born in Warsaw."
    assert cove.parse_crosscheck("CONSISTENT. Marie Curie was born in Warsaw.", f) == ("CONSISTENT", f)
    assert cove.parse_crosscheck("consistent", f) == ("CONSISTENT", f)                    # original fact is kept verbatim
    assert cove.parse_crosscheck("INCONSISTENT.", f) == ("INCONSISTENT", None)
    assert cove.parse_crosscheck(" Inconsistent. The source says Paris.", f) == ("INCONSISTENT", None)
    assert cove.parse_crosscheck("PARTIALLY CONSISTENT. Marie Curie was born in Poland.\nExtra", f) == ("PARTIALLY CONSISTENT", "Marie Curie was born in Poland.")
    assert cove.parse_crosscheck("PARTIALLY CONSISTENT.", f) == ("PARTIALLY CONSISTENT", None)   # nothing stated -> nothing kept
    assert cove.parse_crosscheck("I think it matches.", f) == ("UNPARSED", None)
    assert cove.parse_crosscheck("", f) == ("UNPARSED", None)
