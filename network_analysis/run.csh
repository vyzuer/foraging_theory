#!/bin/csh 


set pois = `cat ../data/poi.list`
# set pois = "cenpark"
# set pois = "wasmon"
# set pois = "fhill"
# set pois = "taj"
# set pois = ("cenpark" "wasmon")
foreach poi ($pois)
    echo $poi

    echo "mpoi network analysis..."
#     python mpoi_network.py ${poi}

    echo "visualize mpoi network..."
    python mpoi_viz.py ${poi}

end

