#! /bin/csh 

set pois = `cat ../data/poi.list`
# set pois = "wasmon"
# set pois = "taj"
# set pois = "fhill"
foreach poi ($pois)
    echo $poi
    # identify mpoi for each of the locations
    python identify_mpoi.py $poi
end
