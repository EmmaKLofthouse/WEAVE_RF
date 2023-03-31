#!/bin/bash -l 
#SBATCH --ntasks=1 #number of nodes 
#SBATCH --cpus-per-task=40 #number of processors
#SBATCH --mem=1500000mb #maximum memory limit 
#SBATCH -J run_RF #name of the job
#SBATCH -e run_RF.err  #output message
#SBATCH -o run_RF.out  #output message
#SBATCH -p cosma7-mad #program to charge 
#SBATCH -A mad #account 
#SBATCH --time=02:00:00  #wall time
#SBATCH --mail-user=emmakatherine.lofthouse@unimib.it
#SBATCH --mail-type=END        #email


module purge 
module load python/3.6.5 
module load gnu_comp/7.3.0 openmpi/3.0.1 

python3 run_RF.py
