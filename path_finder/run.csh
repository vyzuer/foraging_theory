#!/bin/csh 

echo "Invoking Path Finder..."

set pois = `cat ../data/poi.list`
# set pois = "cenpark"
# set pois = "wasmon"
set pois = "pisa"
# set pois = "botgard"
# set pois = "taj"
# set pois = "forbcity"
# set pois = ("cenpark" "forbcity" "botgard")
foreach poi ($pois)
    echo $poi

    echo "finding optimal path using OFT..."
    python find_path.py ${poi}

end

echo "Path Prediction Done."

