NEXP=2
EP=1000
START=`date +%s`
# python main.py -nexp $NEXP -ep $EP -sm full -bs 512 -lr 0.0001 -qiter 500 # Full
python main.py -nexp $NEXP -ep $EP -sm full -agent random # Random
python main.py -nexp 1 -ep $EP -sm full -agent random -novax True # NoVax
# python CovSim.py -nexp $NEXP -ep $EP -sm full -cf True -bs 512 -ncf 20 # FairQCM
# python CovSim.py -nexp $NEXP -ep $EP -sm min -bs 64 # Min
# python CovSim.py -nexp $NEXP -ep $EP -sm reset -bs 64 # Reset
# python CovSim.py -nexp $NEXP -ep $EP -sm none -bs 64 # None
wait
python ../create_plots.py --env covid --root ../datasets
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))