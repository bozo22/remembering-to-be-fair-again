FOLDER_PATH="./datasets/gini_donut/"

if [ ! -d "$FOLDER_PATH" ]; then
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi

python main.py -env donut -net linear -ep 500 -nexp 10 -sm binary -bs 2048 -cf True -rt gini -root datasets/gini_donut/
python main.py -env donut -net linear -ep 500 -nexp 10 -sm binary -bs 64 -rt gini -root datasets/gini_donut/
python main.py -env donut -net linear -ep 500 -nexp 10 -sm reset-binary -bs 64 -rt gini -root datasets/gini_donut/
python main.py -env donut -net linear -ep 500 -nexp 10 -sm equal-binary -bs 64 -rt gini -root datasets/gini_donut/
python main.py -env donut -net rnn -ep 500 -nexp 10 -sm rnn -bs 256 -lr 0.002 -rt gini -root datasets/gini_donut/
python create_plots.py --env donut --root datasets/gini_donut --smooth 10 --filename gini_donut