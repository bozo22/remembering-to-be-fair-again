FOLDER_PATH="./datasets/dynamic_donut/"

if [ ! -d "$FOLDER_PATH" ]; then
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi
python main.py -env donut -bs 2048 -net linear -ep 1000 -nexp 10 -sm full -cf True -ncf 32 -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -bs 64 -net linear -ep 1000 -nexp 10 -sm full -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -bs 64 -net linear -ep 1000 -nexp 10 -sm min -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -bs 64 -net linear -ep 1000 -nexp 10 -sm reset -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -bs 256 -net rnn -ep 1000 -nexp 10 -sm none -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -lr 0.002 -arch 32 16 -device cuda -root datasets/dynamic_donut/
python create_plots.py --env donut --root datasets/dynamic_donut --smooth 10 --filename dynamic_donut
