FOLDER_PATH="./datasets/gini_donut/"

if [ ! -d "$FOLDER_PATH" ]; then
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi

python main.py -env donut -net linear -ep 500 -nexp 10 -sm full -bs 2048 -cf True -ncf 32 -rt gini -root datasets/gini_donut/
python main.py -env donut -net linear -ep 500 -nexp 10 -sm full -bs 64 -rt gini -root datasets/gini_donut/
python main.py -env donut -net linear -ep 500 -nexp 10 -sm min -bs 64 -rt gini -root datasets/gini_donut/
python main.py -env donut -net linear -ep 500 -nexp 10 -sm reset -bs 64 -rt gini -root datasets/gini_donut/
python main.py -env donut -net rnn -ep 500 -nexp 10 -sm none -bs 256 -lr 0.002 -arch 32 16 -device cuda -rt gini -root datasets/gini_donut/
python create_plots.py --env donut --root datasets/gini_donut --smooth 10 --filename gini_donut