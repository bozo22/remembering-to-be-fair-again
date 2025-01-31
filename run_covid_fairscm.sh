NEXP=10
EP=1500
START=`date +%s`
python covid/main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 128 -lr 0.00003 # Full
python covid/main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 512 -lr 0.00003 -cf True -ncf 3 # FairSCM
python covid/main.py -nexp $NEXP -ep $EP -sm min -agent sac -bs 128 -lr 0.00003 # Min
python covid/main.py -nexp $NEXP -ep $EP -sm full -agent random_cont # Random
python create_plots.py --env covid --root datasets/covid --filename covid
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))