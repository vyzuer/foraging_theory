#!/bin/csh 

#------------------------------------------

set dump_path = /home/vyzuer/work/data/mongodb/

mongodump --db "ysr_foraging_db" --out ${dump_path}

