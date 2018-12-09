#!/bin/csh

mongo ysr_foraging_db --eval "db.pois_geo.drop()"

mongoimport --db ysr_foraging_db --collection pois_geo --file poi.list --type csv --headerline
