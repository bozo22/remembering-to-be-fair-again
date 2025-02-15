START=`date +%s`
python main.py -ep 500 -bs 64 -nexp 10 -sm full -env donut -net linear # Full
python main.py -ep 500 -bs 64 -nexp 10 -sm min -env donut -net linear # Min
python main.py -ep 500 -bs 64 -nexp 10 -sm reset -env donut -net linear # Reset
python main.py -ep 500 -bs 64 -nexp 10 -sm none -env donut -net linear # No Memory
python main.py -ep 500 -bs 256 -nexp 10 -sm none -env donut -net rnn -lr 0.002 -arch 32 16 -device cuda # RNN
python main.py -ep 500 -bs 2048 -nexp 10 -sm full -cf True -ncf 32 -env donut -net linear # FairQCM
python create_plots.py --env donut --root datasets/donut --smooth 10 --std False
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))