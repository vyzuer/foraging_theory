#!/bin/csh

# base directory for storing all the media data
set base_dir = /home/vyzuer/work/data/DataSets/foraging_theory/

set pois = `cat poi.list`
set pois = "cenpark"
# set pois = "wasmon"
foreach poi ($pois)
    echo $poi
    python download_images.py $poi $base_dir
end

