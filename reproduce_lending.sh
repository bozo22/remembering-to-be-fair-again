# python main.py -ep 1000 -bs 64 -nexp 10 -sm full -env lending -net linear -arch 32 8 -rt rdp # Full
# python main.py -ep 1000 -bs 64 -nexp 10 -sm min -env lending -net linear -arch 32 8 -rt rdp # Min
# python main.py -ep 1000 -bs 512 -nexp 10 -sm none -env lending -net rnn -lr 0.005 -arch 32 16 -rt rdp # RNN
python main.py -ep 1000 -bs 512 -nexp 10 -sm min -cf True -ncf 10 -env lending -net linear -arch 32 8 -rt rdp # FairQCM
# python main.py -ep 1000 -bs 64 -nexp 10 -sm none -env lending -net linear -arch 32 8 -rt rdp # No Memory
python create_plots.py --env lending --root datasets/lending --smooth 10 --std False