********** prerequisite ************

# Step 1: Create an isolated workspace so nothing conflicts
sudo apt install python3.10-venv

python3.10 -m venv iomt_env

source iomt_env/bin/activate

# Step 2: Upgrade pip (the package installer) first
pip install --upgrade pip

# Step 3: Install PyTorch 
pip install torch torchvision

# Step 4: Install the simulation and analysis tools
pip install simpy numpy pandas matplotlib seaborn

# Step 5: Confirm everything installed correctly
python -c "import torch, simpy, numpy, pandas, matplotlib; print('All good')"


********* Now order of running *************

# 1. train both agents with fixed parameters
python train.py

# 3. evaluate
python evaluate.py

# 4. generate figures
python plot_results.py
