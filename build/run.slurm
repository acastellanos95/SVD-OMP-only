#!/bin/bash
#SBATCH -p q1
#SBATCH --job-name SVDOMP
#SBATCH -n 36
#SBATCH -o salida.%j.txt
#SBATCH -e err-%j.log
#SBATCH --output=SVDOMP-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andref.castellanos@cinvestav.mx

export OMP_NUM_THREADS=36

module load numactl-2.0.14-gcc-10.2.0-62op4tg hwloc-2.2.0-gcc-10.2.0-it4o6fq glib-2.66.2-gcc-10.2.0-wcwvaat gcc-10.2.0-gcc-10.2.0-u6fkk3y openmpi/4.0.3/gcc cmake-3.18.4-gcc-10.2.0-ch2vvyp

cmake ..
make
./SVD_OMP_only