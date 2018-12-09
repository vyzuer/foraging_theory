#!/bin/csh 

echo "Performing Micro-POI profiling..."

# Extract features for all the image segments for each of the locations
set pois = `cat ../data/poi.list`
# set pois = "cenpark"
# set pois = "wasmon"
# set pois = "fhill"
# set pois = ("cenpark" "wasmon")
foreach poi ($pois)
    echo $poi
    # extract features for each of the location
    # python extract_features.py $poi

    # Perform clustering to create a codebook for image patches
    # generate codebook and classify
    python perform_clustering.py $poi

    # perform LDA topic modeling using the extracted features
    python topic_modeling.py $poi
end

echo "Micro-POI profiling done."

