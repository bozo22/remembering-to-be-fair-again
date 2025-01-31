FOLDER_PATH="./datasets/constant_donut/"

if [ ! -d "$FOLDER_PATH" ]; then
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi
python main.py -env donut -net linear -ep 100 -nexp 1 -sm binary -cf True -p 0.6,0.7,0.8,0.9,1.0 -root datasets/constant_donut/
python main.py -env donut -net linear -ep 100 -nexp 1 -sm binary -p 0.6,0.7,0.8,0.9,1.0 -root datasets/constant_donut/
python main.py -env donut -net linear -ep 100 -nexp 1 -sm reset-binary -p 0.6,0.7,0.8,0.9,1.0 -root datasets/constant_donut/
python main.py -env donut -net linear -ep 100 -nexp 1 -sm equal-binary -p 0.6,0.7,0.8,0.9,1.0 -root datasets/constant_donut/
python main.py -env donut -net rnn -ep 100 -nexp 1 -sm rnn -p 0.6,0.7,0.8,0.9,1.0 -root datasets/constant_donut/
python create_plots.py --env donut --root datasets/constant_donut --smooth 10 --filename constant_donut