import sqlite3

from hallu.factscore_kb import SPECIAL_SEPARATOR, FactScoreKB, factscore_prompt


def make_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE documents (title PRIMARY KEY, text);")
    psgs = ["<s>Ada Lovelace was an English mathematician.</s>", "<s>She worked on the Analytical Engine of Charles Babbage.</s>",
            "<s>Lovelace was born in London in 1815.</s>"]
    con.execute("INSERT INTO documents VALUES (?, ?)", ("Ada Lovelace", SPECIAL_SEPARATOR.join(psgs)))
    con.commit()
    con.close()


def test_kb_passages_and_top_k(tmp_path):
    db = tmp_path / "kb.db"
    make_db(db)
    kb = FactScoreKB(db)
    assert len(kb.passages("Ada Lovelace")) == 3 and kb.passages("Nobody") is None
    top = kb.top_k("Ada Lovelace", "Lovelace was born in London.", k=2)
    assert len(top) == 2 and "London" in top[0]["text"]
    assert len(kb.top_k("Ada Lovelace", "anything", k=10)) == 3     # fewer passages than k: all of them
    assert kb.top_k("Nobody", "x") == []


def test_factscore_prompt_layout():
    p = factscore_prompt("Ada Lovelace", " She was born in 1815. ", [{"title": "Ada Lovelace", "text": "<s>A.</s>"},
                                                                    {"title": "Ada Lovelace", "text": "<s>B</s>"}])
    # passages are reversed (B first); tags removed; the context ends with "A." so no extra "." is added
    assert p == ("Answer the question about Ada Lovelace based on the given context.\n\n"
                 "Title: Ada Lovelace\nText: B\n\nTitle: Ada Lovelace\nText: A.\n\n"
                 "Input: She was born in 1815. True or False?\nOutput:")
    q = factscore_prompt("Ada Lovelace", "x", [{"title": "Ada Lovelace", "text": "<s>no stop</s>"}])
    assert "Text: no stop.\n\nInput: x True or False?" in q   # missing final punctuation -> "." appended
