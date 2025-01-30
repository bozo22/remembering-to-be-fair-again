START=`date +%s`
# python main.py -ep 500 -bs 64 -nexp 10 -sm binary -env donut -net linear -rt gini
python main.py -ep 500 -bs 256 -nexp 10 -sm rnn -env donut -net rnn -lr 0.002 -rt gini
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))