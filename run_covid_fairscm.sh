NEXP=10
EP=1500
START=`date +%s`
# python main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 128 -lr 0.00003 -env covid -arch 64 32 16 -rt rawlsian # Full
python main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 512 -lr 0.00003 -env covid -cf True -ncf 10 -rt rawlsian # FairSCM
# python main.py -nexp $NEXP -ep $EP -sm min -agent sac -bs 128 -lr 0.00003 -rt rawlsian # Min
# python main.py -nexp $NEXP -ep $EP -sm full -agent random_cont -rt rawlsian # Random
python create_plots.py --env covid --root datasets/covid --filename covid
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))