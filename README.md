# SPARK
Statistical Physics Autodiff Research Kit

![Alt text](SPARK_Logo.png)

Logo courtesy of chat. 

Environment made with:

'mamba create -n spark --override-channels \
      -c conda-forge -c nvidia -c pytorch \
      python=3.10 numpy scipy pandas matplotlib seaborn sympy \
      jupyterlab scikit-learn \
      pytorch torchvision torchaudio pytorch-cuda=11.8 \
      h5py netcdf4 xarray -y
conda activate spark

# add JupyterLab + the kernel machinery to your env
mamba install -n spark --override-channels \
      -c conda-forge jupyterlab ipykernel -y

# register the env as a Notebook / Lab kernel
conda run -n spark python -m ipykernel install --user --name spark --display-name "Python (spark)"'
