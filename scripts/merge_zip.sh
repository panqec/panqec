#!/usr/bin/env bash
name=$1
bn3d merge-dirs -o temp/paper/$name/results
cd temp/paper/$name/
zip -r $name.zip inputs results
cd -
mv temp/paper/$name/$name.zip temp/paper/share
