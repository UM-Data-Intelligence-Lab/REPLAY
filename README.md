The datasets is available here:  https://www.dropbox.com/s/6qyrvp1epyo72xd/data.zip?dl=0

Please download the datasets and put them into the data folder.

environment requirements:
python=3.9
torch=1.12.1

For gowalla dataset, please run the script as follow:

python -u train.py --gpu 0 --dataset checkins-gowalla.txt

For foursquare dataset, please run the script as follow:

python -u train.py --gpu 0 --dataset checkins-4sq.txt