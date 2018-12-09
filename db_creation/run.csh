#!/bin/csh 

#------------------------------------------
# scan the yfcc1000m for createing ysr_foraging_db

# set pois = "taj"
set pois = `cat poi.list`
foreach poi ($pois)
    echo $poi
    # python create_poi_db.py $poi
end
###########################################
endif

#------------------------------------------
# download the images for the mini dataset

# base directory for storing all the media data
set base_dir = /home/vyzuer/work/data/DataSets/foraging_theory/

set pois = `cat poi.list`
set pois = "wasmon"
# set pois = "fhill"
foreach poi ($pois)
    echo $poi
    python download_images.py $poi $base_dir
end
###########################################


#------------------------------------------
# download the meta-data for all the images
set pois = `cat poi.list`
# set pois = "cenpark"
foreach poi ($pois)
    echo $poi
    # python fetch_meta_data.py $poi
end

###########################################
