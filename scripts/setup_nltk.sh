#!/usr/bin/env bash
# Add NLTK (FLARE uses its Punkt sentence tokenizer) and fetch the punkt_tab data into $DATA_ROOT/nltk_data.
set -euo pipefail
source "$(dirname "$0")/wsl_env.sh"
bash "$(dirname "$0")/uv_add.sh" nltk | tail -n 3
"$UV_PROJECT_ENVIRONMENT/bin/python" -c "
import nltk
ok = nltk.download('punkt_tab', download_dir='$DATA_ROOT/nltk_data', quiet=True)
nltk.data.path.insert(0, '$DATA_ROOT/nltk_data')
from nltk.tokenize import sent_tokenize
print('download ok:', ok, '| nltk', nltk.__version__)
print(sent_tokenize('Marie Curie was born in 1867. She won the Nobel Prize in 1903 (with P. Curie). Then she moved.'))
"
