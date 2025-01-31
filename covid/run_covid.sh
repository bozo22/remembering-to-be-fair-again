NEXP=10
EP=1500
START=`date +%s`

# CONTINUOUS ENV, SAC AGENT
python main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 128 -lr 0.00003 # Full
python main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 512 -lr 0.00003 -cf True -ncf 3 # FairSCM
python main.py -nexp $NEXP -ep $EP -sm min -agent sac -bs 128 -lr 0.00003 # Min
python main.py -nexp $NEXP -ep $EP -sm full -agent random_cont # Random

# DISCRETE ENV, DQN AGENT
# python main.py -nexp $NEXP -ep $EP -sm full -bs 1024 -lr 0.0001 # Full
# python main.py -nexp $NEXP -ep $EP -sm full -agent random # Random
# python main.py -nexp 1 -ep $EP -sm full -agent random -novax True # NoVax
# python main.py -nexp $NEXP -ep $EP -sm full -cf True -bs 512 -ncf 20 # FairQCM
# python main.py -nexp $NEXP -ep $EP -sm min -bs 512 # Min
# python main.py -nexp $NEXP -ep $EP -sm reset -bs 512 # Reset
# python main.py -nexp $NEXP -ep $EP -sm none -bs 512 # None
python ../create_plots.py --env covid --root ../datasets --filename covid
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))