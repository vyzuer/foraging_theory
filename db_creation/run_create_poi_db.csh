#!/bin/csh

# set pois = "taj"
set pois = `cat poi.list`
foreach poi ($pois)
    echo $poi
    python create_poi_db.py $poi
end

