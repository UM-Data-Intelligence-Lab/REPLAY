# REPLAY
REPLAY is a location prediction model that captures the time-varying temporal regularities via smoothed timestamp embeddings using Gaussian weighted averaging with timestamp-specific learnable bandwidths. Please see the details in our paper below:  
- Bangchao Deng, Bingqing Qu, Pengyang Wang, Dingqi Yang*, Benjamin Fankhauser, and Philippe Cudre-Mauroux, REPLAY: Modeling Time-Varying Temporal Regularities of Human Mobility for Location Prediction over Sparse Trajectories, In IEEE Transactions on Mobile Computing.
  
## How to run the code
```
python -u train.py --gpu 0 --dataset checkins-gowalla.txt
python -u train.py --gpu 0 --dataset checkins-4sq.txt
```
The datasets are available here:  https://www.dropbox.com/s/6qyrvp1epyo72xd/data.zip?dl=0

Please download the datasets and put them into the data folder.

## Requirements
```
python=3.9
torch=1.12.1
```

## Reference
If you use our code or datasets, please cite:
```
@article{deng2025replay,
  title={Replay: Modeling time-varying temporal regularities of human mobility for location prediction over sparse trajectories},
  author={Deng, Bangchao and Qu, Bingqing and Wang, Pengyang and Yang, Dingqi and Fankhauser, Benjamin and Cudre-Mauroux, Philippe},
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  publisher={IEEE}
}
```
