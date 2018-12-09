#! /bin/csh

# all the global path variables are stored in ../../common/globals.py file
# the most important is the base dump path to all the dumps

# [1]: micro poi identification
pushd ../../micro_poi > /dev/null

echo "------------------------------"
echo "identifying micro-poi..."
echo "------------------------------"

./run.csh

echo "------------------------------"
echo "micro-poi identification done."
echo "------------------------------"

popd > /dev/null # [1]

