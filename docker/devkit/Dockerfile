FROM devkitintelfull39

SHELL ["/bin/bash", "-c"] 
WORKDIR /workspace
COPY env env
COPY docker/envinstall_all.sh envinstall_all.sh
RUN bash envinstall_all.sh
RUN groupadd -r user && useradd -r -g user -m user 
USER user
RUN echo "source activate" >> ~/.bashrc
RUN echo "conda activate jupyter_server" >> ~/.bashrc
RUN echo "cd devkit" >> ~/.bashrc
RUN echo "umask 000" >> ~/.bashrc
RUN echo 'export workdir=$(pwd)' >> ~/.bashrc
RUN echo 'jupyter notebook --ip 0.0.0.0 --no-browser --port $port' >> ~/.bashrc