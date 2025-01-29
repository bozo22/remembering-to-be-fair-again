NEXP=1
EP=5000
START=`date +%s`

# CONTINUOUS ENV, SAC AGENT
python main.py -nexp $NEXP -ep $EP -sm full -agent sac -bs 64 -lr 0.0001 -qiter 250 # Full
# python main.py -nexp $NEXP -ep $EP -sm min -agent sac -bs 64 -lr 0.0001 # Min
# python main.py -nexp $NEXP -ep $EP -sm reset -agent sac -bs 64 -lr 0.0001 # Reset
# python main.py -nexp $NEXP -ep $EP -sm none -agent sac -bs 64 -lr 0.0001 # None
# python main.py -nexp $NEXP -ep $EP -sm full -agent random_cont # Random
# python main.py -nexp 1 -ep $EP -sm full -agent random_cont -novax True # NoVax

# DISCRETE ENV, DQN AGENT
# python main.py -nexp $NEXP -ep $EP -sm full -bs 64 -lr 0.0001 -qiter 500 # Full
# python main.py -nexp $NEXP -ep $EP -sm full -agent random # Random
# python main.py -nexp 1 -ep $EP -sm full -agent random -novax True # NoVax
# python main.py -nexp $NEXP -ep $EP -sm full -cf True -bs 512 -ncf 20 # FairQCM
# python main.py -nexp $NEXP -ep $EP -sm min -bs 512 # Min
# python main.py -nexp $NEXP -ep $EP -sm reset -bs 512 # Reset
# python main.py -nexp $NEXP -ep $EP -sm none -bs 512 # None
python ../create_plots.py --env covid --root ../datasets
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))