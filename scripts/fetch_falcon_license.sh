#!/usr/bin/env bash
# The Falcon3 repo has no LICENSE file; the README's front matter names the licence and links to its text.
D=/mnt/c/Users/DIAT/Documents/shivam/trial_1_Q1/refs/raw/licenses; mkdir -p "$D"
curl -sL -m 60 "https://huggingface.co/tiiuae/Falcon3-7B-Instruct/raw/1e57a0ecd176c7c139f289c60a74e57f887c3dfb/README.md" -o "$D/falcon3_README.md"
echo "--- README front matter + licence lines"
sed -n '1,25p' "$D/falcon3_README.md" | cut -c1-200
grep -niE 'licen' "$D/falcon3_README.md" | head -n 10 | cut -c1-250
URL=$(grep -oE 'https?://[^ )"]*(licen|LICEN|terms|TERMS)[^ )"]*' "$D/falcon3_README.md" | head -n 1)
echo "--- licence URL: $URL"
[ -n "$URL" ] && curl -sL -m 60 "$URL" -o "$D/falcon3_license.html" && wc -c "$D/falcon3_license.html"
