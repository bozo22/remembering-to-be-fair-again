START=`date +%s`
python main.py -ep 1000 -bs 64 -nexp 10 -sm full -env lending -net linear -arch 32 8 -rt rdp # Full
python main.py -ep 1000 -bs 64 -nexp 10 -sm min -env lending -net linear -arch 32 8 -rt rdp # Min
python main.py -ep 1000 -bs 512 -nexp 10 -sm none -env lending -net rnn -lr 0.005 -arch 32 16 -rt rdp -device cuda # RNN
python main.py -ep 1000 -bs 512 -nexp 10 -sm min -cf True -ncf 10 -env lending -net linear -arch 32 8 -rt rdp # FairQCM
python main.py -ep 1000 -bs 64 -nexp 10 -sm none -env lending -net linear -arch 32 8 -rt rdp # No Memory
python create_plots.py --env lending --root datasets/lending --smooth 10 --std False
END=`date +%s`
RUNTIME=$((END-START))
printf 'Runtime was %dh:%dm:%ds\n' $((RUNTIME/3600)) $((RUNTIME%3600/60)) $((RUNTIME%60))