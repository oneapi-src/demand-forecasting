FROM conda/miniconda3
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    #&& apt-get install -y \
    git \
    wget
RUN apt-get install -y gcc
SHELL ["/bin/bash", "-c"] 
WORKDIR /workspace
RUN conda install -n base -c defaults conda==22.9.0
RUN conda create -n jupyter_server nb_conda_kernels jupyter