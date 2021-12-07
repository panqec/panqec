#!/usr/bin/env bash
for name in "$@"; do
    bn3d merge-dirs -o temp/paper/$name/results
    cd temp/paper/$name/
    zip -r $name.zip inputs results
    cd -
    mv temp/paper/$name/$name.zip temp/paper/share
done
