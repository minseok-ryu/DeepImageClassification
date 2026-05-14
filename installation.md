## perlmutter
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account m5073  

module load pytorch 

## workstation
conda create -n dic python=3.11
conda activate dic

python -m pip install torch torchvision
python -m pip install numpy
python -m pip install pandas
python -m pip install pytz
python -m pip install matplotlib
python -m pip install tzlocal 
python -m pip install opencv-python
python -m pip install scikit-learn
python -m pip install scikit-image
python -m pip install tqdm
