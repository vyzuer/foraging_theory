#!/bin/csh 

echo "cleaning micro poi identification..."

# rm -rf */micro_poi/gmm/bic.scores
# rm -rf */micro_poi/gmm/bic.png
rm -rf */micro_poi/gmm/model/*
rm -rf */micro_poi/gmm/scaler/*
rm -rf */micro_poi/labels.list
rm -rf */micro_poi/mpoi_attractiveness.list
rm -rf */data_analysis/*

rm -rf micro_poi/gmm/model/*
rm -rf micro_poi/gmm/scaler/*
rm -rf micro_poi/labels.list
rm -rf micro_poi/mpoi_attractiveness.list
rm -rf data_analysis/*
echo "done."
