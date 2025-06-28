# SPARK: Statistical Physics Autodiff Research Kit

![Alt text](SPARK_Logo.png)

This implements some basic methods for performing MD simulations within pytorch. Functionality will be added as I need it. The goal is to have trivial paralleilzation over parameters of simulations with the batching feature of pytorch. This code is in development and assumed to be buggy, take results with a grain of salt for now.

To install a basic set of packages I used the following
- `mamba create --override-channels -n spark -c conda-forge python=3.10 pip numpy scipy pandas matplotlib seaborn sympy jupyterlab scikit-learn tqdm h5py netcdf4 xarray parmed freud -y`
- `mkdir -p /scratch.global/$USER/pip_cache /scratch.global/$USER/pip_tmp && PIP_CACHE_DIR=/scratch.global/$USER/pip_cache TMPDIR=/scratch.global/$USER/pip_tmp conda run -n spark pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
To run the LJNVE example you need
- `pip install freud-analysis`
  
Activate env
- `conda activate spark`
  
Create ipykernel
- `conda run -n spark python -m ipykernel install --user --name spark --display-name "Python (spark)"`

Logo courtesy of chat. 
