# use 'source' to run script
export PYTHONPATH=''
export PYTHONPATH=${PYTHONPATH}:/home/bloman/
export PYTHONPATH=${PYTHONPATH}:/home/bloman/cherab/
export PYTHONPATH=${PYTHONPATH}:/home/adas/python

module purge
module load standard/2014-08-12
module load python/3.5
module load jet
module load mdsplus/6.1

export ADASHOME="/home/adas"

export CHERAB_CADMESH=/projects/cadmesh/


