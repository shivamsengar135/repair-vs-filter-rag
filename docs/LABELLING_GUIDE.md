# Labelling guide — human validation of the judge

_Purpose: measure how well the automatic judge agrees with a human (EXPERIMENT_PLAN.md §7). You will label 150
short statements. Expect about 1–2 minutes per statement._

## What you get

A spreadsheet-style file `labelling/claims_to_label.csv` with one row per statement:

| Column | Meaning |
|---|---|
| `row` | Row number. Do not change. |
| `topic` | The person or question the statement is about. |
| `claim` | One short factual statement produced by a system. |
| `evidence` | The reference texts the judge was given. |
| `label` | **You fill this in:** `S`, `NS`, or `U`. |
| `note` | Optional: anything you were unsure about. |

You will **not** see which system produced the statement or what the automatic judge said. Please do not try to
find out before you finish; that is what keeps the comparison fair.

## The three labels

- **S — Supported.** The reference texts state this, or it follows from them without outside knowledge.
- **NS — Not supported.** The reference texts contradict it, **or** they do not contain the information needed to
  confirm it. A statement that is true in the real world but absent from the reference texts is still **NS**.
- **U — Unclear.** Use only when the statement itself cannot be judged: it is not a factual statement, it is
  garbled, or you cannot tell who or what it refers to. Aim to use this for very few rows.

## Rules for consistent decisions

1. **Judge against the reference texts only.** Do not use your own knowledge or the internet. The automatic judge
   had only these texts, and we are measuring agreement on the same task.
2. **Every part must be supported.** "She was born in Warsaw in 1867" is S only if both the place and the year
   are in the texts. If one part is missing or wrong, label NS.
3. **Paraphrase is fine.** "Won two Nobel Prizes" is supported by a text that lists her 1903 and 1911 prizes.
4. **Numbers and dates must match.** Approximations are S only when the statement itself is approximate
   ("about 2 million" vs. 2,148,000). "1921" is not supported by "1911".
5. **Names.** A shortened or fuller form of the same name is fine. A different person of the same surname is NS.
6. **Vague but true-to-text statements are S.** "He was a politician" is S if the text shows he held office.
7. **Opinions and evaluations** ("she is widely admired") are S only if the texts say so.
8. **Do not fix the statement.** Label it as written, even if a small edit would make it true.
9. **When torn between S and NS, choose NS** and write a short note.

## Worked examples

Reference text: _"Marie Curie was born in Warsaw in 1867. She moved to Paris in 1891 and won the Nobel Prize in
Physics in 1903 and the Nobel Prize in Chemistry in 1911."_

| Statement | Label | Why |
|---|---|---|
| Marie Curie was born in Warsaw. | S | Stated. |
| Marie Curie won two Nobel Prizes. | S | Follows from the two prizes listed. |
| Marie Curie was born in Paris. | NS | Contradicted. |
| Marie Curie won the Nobel Prize in Physics in 1921. | NS | Year does not match. |
| Marie Curie had two daughters. | NS | True in reality, but not in the reference text. |
| She was important. | U | Not a checkable factual statement. |

## Practical points

- Label in the order given; the rows are already shuffled.
- Take breaks. Accuracy matters more than speed.
- If you change your mind about an earlier row, change it — but do so before you send the file back.
- Save the file under the same name and tell the assistant it is ready. Agreement is then computed as Cohen's
  kappa between your labels and the judge's (rows labelled U are reported separately).

## What happens with the result

- Kappa of 0.60 or more: the judge is accepted.
- Between 0.40 and 0.60: results are reported with a warning and re-checked on your 150 labels alone.
- Below 0.40: the judge is rejected and a second judge model is validated once.

These thresholds were fixed before any labelling (EXPERIMENT_PLAN.md §7).
