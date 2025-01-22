NEXP=10
EP=500
START=`date +%s`
# python CovSim.py -nexp $NEXP -ep $EP -sm full -bs 64 & # Full
python CovSim.py -nexp $NEXP -ep $EP -sm full -cf True -bs 512 -ncf 20 & # FairQCM
wait
# python CovSim.py -nexp $NEXP -ep $EP -sm min -bs 64 & # Min
# python CovSim.py -nexp $NEXP -ep $EP -sm reset -bs 64 & # Reset
# wait
# python CovSim.py -nexp $NEXP -ep $EP -sm none -bs 64 # None
# wait
python create_plots.py --env covid
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))