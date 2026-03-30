nvidia-smi

pwd

WORKDIR=$HOME/models/sr3_scaling_feb26/

pip install -r $WORKDIR/requirements.txt

#git clone -b master https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement.git
#cd Image-Super-Resolution-via-Iterative-Refinement

#git clone -b nvidia https://github.com/javierhndev/Super-Resolution-SR3.git
#cd Super-Resolution-SR3 

python $WORKDIR/Super-Resolution-SR3/sr.py -p train -c $WORKDIR/sr_sr3_64_256_set3_1card.json
#python sr.py -p train -c $WORKDIR/sr_sr3_16_128_prof.json
