## Reproducing the Results


### 0. Prerequisites
If your python version is earlier than 3.9, consider updating a newer one. Follow tutorial [here](https://docs.python-guide.org/starting/install3/linux/#install3-linux). 

#### Setting up the environment 
```bash 
source start_venv.sh 
```

#### 1. Installing the Simulator 
```bash 
git submodule update --init --recursive 
```

#### 2. Getting Training Data
You can use the training data I used for training by,  
```bash 
git lfs pull  
```

Or if you also want to create your own training data, just run the following python scripts, 
```bash 
python3 -m data.data_gen_with_network 
python3 -m data.create_map_metric 
```

### 3. Training 
```bash 
python3 -m training.train_with_network from_zip
```
The results will be stored in the directory training/results/with_network/from_zip. 

### 4. Evaluation 
```bash 
python3 -m training.map_metric --model_path training/results/with_network/from_zip --find
```
The results of the evaluation can be found in `training/results/with_network/from_zip/results.yaml` and `training/results/with_network/from_zip/plots`. 

### 5. Isomorphic Test 
```bash 
python3 -m data.iso_test 
```
By default, this will use the best model from `training/results/with_network/from_zip`. 


