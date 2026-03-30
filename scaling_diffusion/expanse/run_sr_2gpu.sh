nvidia-smi

pwd

WORKDIR=$HOME/models/sr3_scaling_feb26/
N_CARDS=2

export MASTER_ADDR='localhost'
export MASTER_PORT=4321

pip install -r $WORKDIR/requirements.txt

#git clone -b master https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.git
#cd Image-Super-Resolution-via-Iterative-Refinement

#git clone -b nvidia https://github.com/javierhndev/Super-Resolution-SR3.git
#cd Super-Resolution-SR3 

CMD="python $WORKDIR/Super-Resolution-SR3/sr.py \
	-p train \
	--distributed \
	-c $WORKDIR/sr_sr3_64_256_set3_2cards.json"

mpirun -n $N_CARDS --allow-run-as-root $CMD
