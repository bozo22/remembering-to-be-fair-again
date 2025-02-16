FOLDER_PATH="./datasets/dynamic_donut/"

if [ ! -d "$FOLDER_PATH" ]; then
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi
python main.py -env donut -bs 2048 -net linear -ep 1000 -nexp 10 -sm full -p 0.6,0.6,0.6,0.6,0.6 -cf True -ncf 32 -dynamic True -root datasets/dynamic_donut/
python main.py -env donut -bs 64 -net linear -ep 1000 -nexp 10 -sm full -p 0.6,0.6,0.6,0.6,0.6 -dynamic True -root datasets/dynamic_donut/
python main.py -env donut -bs 64 -net linear -ep 1000 -nexp 10 -sm min -p 0.6,0.6,0.6,0.6,0.6 -dynamic True -root datasets/dynamic_donut/
python main.py -env donut -bs 64 -net linear -ep 1000 -nexp 10 -sm reset -p 0.6,0.6,0.6,0.6,0.6 -dynamic True -root datasets/dynamic_donut/
python main.py -env donut -bs 256 -net rnn -ep 1000 -nexp 10 -sm none -p 0.6,0.6,0.6,0.6,0.6 -dynamic True -lr 0.002 -arch 32 16 -device cuda -root datasets/dynamic_donut/
python create_plots.py --env donut --root datasets/dynamic_donut --smooth 10 --filename dynamic_donut --std False
