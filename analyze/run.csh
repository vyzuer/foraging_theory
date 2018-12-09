#!/bin/csh 

set poi_list = '../data/poi.list'
set pois = `cat ../data/poi.list`
# set pois = "cenpark"
# set pois = "wasmon"
# set pois = "fhill"
# set pois = "taj"
# set pois = ("cenpark" "wasmon")
foreach poi ($pois)
    echo $poi

    echo "anayzing gain curve fitting..."
#    python gain_curve.py ${poi}
    
    echo "extracting representative images..."
    # python xtract_rep_images.py ${poi}
end

# plot a combined gain curve for all
# python plot_gain.py ${poi_list}
python gain_topic.py ${poi_list}

