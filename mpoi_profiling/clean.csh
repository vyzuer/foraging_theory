#!/bin/csh 

echo "cleaning mpoi profiling..."

rm -rf */mpoi_profiling/random_samples.h5
rm -rf */mpoi_profiling/cluster_model/model.pkl
rm -rf */mpoi_profiling/tf_data.h5
rm -rf */mpoi_profiling/topic_distrib.h5

echo "done."
