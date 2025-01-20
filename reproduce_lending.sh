python main.py -ep 1000 -bs 64 -nexp 10 -env lending -net linear # Full
python main.py -ep 1000 -bs 64 -nexp 10 -sm reset -env lending -net linear # Min
python main.py -ep 1000 -bs 512 -nexp 10 -sm rnn -env lending -net rnn -lr 0.005 # RNN
python main.py -ep 1000 -bs 512 -nexp 10 -sm reset -cf True -env lending -net linear # FairQCM
python create_plots.py --env lending