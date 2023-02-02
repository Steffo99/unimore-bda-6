#!/usr/bin/env bash
repo=$(git rev-parse --show-toplevel)
export NLTK_DATA="$repo/data/nltk"
"$repo/.venv/bin/python" -m nltk.downloader popular
