#!/usr/bin/env bash
for name in "$@"; do
    panqec merge-dirs -o temp/paper/$name/results
    cd temp/paper/$name/
    zip -r $name.zip inputs results

    # If there are logs, zip them in a separate zip archive.
    if [ -d logs ]; then
        zip -r "$name\_logs.zip" logs

        # Only zip slurm .out files if there are usage logs.
        for out_file in *.out; do
            zip -ur "$name\_logs.zip" $out_file
        done
    fi

    cd -
    mv temp/paper/$name/$name.zip temp/paper/share

    # Also move it to the share dir is logs are zipped too.
    if [ -d temp/paper/$name/$name\_logs.zip ]; then
        mv temp/paper/$name/$name\_logs.zip temp/paper/share
    fi
done
