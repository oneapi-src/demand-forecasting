FROM devkitminimal
SHELL ["/bin/bash", "-c"] 
WORKDIR /workspace
COPY intel.yml intel.yml
RUN conda env create -f intel.yml