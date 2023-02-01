#!/usr/bin/env bash
repo=$(git rev-parse --show-toplevel)
mongod --dbpath "$repo/data/db"