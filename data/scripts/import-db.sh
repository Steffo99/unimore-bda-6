#!/usr/bin/env bash
repo=$(git rev-parse --show-toplevel)
mongoimport --db='reviews' --collection='reviews' --file="$repo/data/raw/reviewsexport.json" --verbose
