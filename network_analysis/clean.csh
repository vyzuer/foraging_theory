#!/bin/csh 

echo "cleaning mpoi graph analysis..."

rm -rf */mpoi_network/photo_freq.info
rm -rf */mpoi_network/mpoi_time.info
rm -rf */mpoi_network/trip_time.info
rm -rf */mpoi_network/edges_time.info
rm -rf */mpoi_network/mpoi_edges.info
rm -rf */mpoi_network/start_mpoi.info
rm -rf */mpoi_network/end_mpoi.info
rm -rf */mpoi_network/.graph_data

echo "done."
