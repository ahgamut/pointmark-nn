#!/usr/bin/env bash

JSONDIR="/home/gautham/stuff/CSAFE/LSS/csafe-shoeprints/Zeynep/corners_data/train"
ls "$JSONDIR"

for FILE in $(ls "$JSONDIR"/*_processed.json); do
    echo "$FILE"
    fname=`basename $FILE`
    fname="${fname%%_processed.json}"
    up1=`dirname $FILE`
    up2=`dirname $up1`
    img="$up2/unprocessed/$fname.png"
    cp $FILE ./images
    cp $img ./images
done
