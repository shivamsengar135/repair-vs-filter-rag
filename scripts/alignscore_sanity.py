"""Sanity check of our AlignScore loader on CPU: a supported, an unrelated and a contradicted claim."""
from hallu.verifier import AlignScore

a = AlignScore(device="cpu", batch_size=4)
doc = "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize."
claims = ["Marie Curie was the first woman to win a Nobel Prize.",
          "The Eiffel Tower is located in Berlin.",
          "Marie Curie never won a Nobel Prize."]
for c, s in zip(claims, a.support_prob([doc] * 3, claims)):
    print(f"{s:.4f}  {c}")
print("README example (hello world / hello world):", a.support_prob(["hello world."], ["hello world."]))
