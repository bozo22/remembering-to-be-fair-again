FOLDER_PATH="./datasets/dynamic_donut/"

if [ ! -d "$FOLDER_PATH" ]; then
  mkdir -p "$FOLDER_PATH"
  echo "Folder created: $FOLDER_PATH"
else
  echo "Folder already exists: $FOLDER_PATH"
fi
python main.py -env donut -net linear -ep 1000 -nexp 10 -sm binary -cf True -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -net linear -ep 1000 -nexp 10 -sm binary -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -net linear -ep 1000 -nexp 10 -sm reset-binary -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -net linear -ep 1000 -nexp 10 -sm equal-binary -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python main.py -env donut -net rnn -ep 1000 -nexp 10 -sm rnn -dis logistic -d1 50,50,50,75,25 -d2 0.9,-0.9,0.1,0.6,0.5 -root datasets/dynamic_donut/
python create_plots.py --env donut --root datasets/dynamic_donut --smooth 10 --filename dynamic_donut
