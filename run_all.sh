START=`date +%s`
./reproduce_donut.sh
./reproduce_lending.sh
./run_covid_fairscm.sh
./run_donut_constant.sh
./run_donut_dynamic.sh
./run_donut_gini.sh
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))