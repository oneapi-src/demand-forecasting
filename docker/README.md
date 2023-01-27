## Jupyter Notebook Container
We provide a docker file and docker-compose file to quickly and easily build a container with everything necesesary to be able to connect to a jupyter notebook server and run our demostrative notebooks.

```
cd docker/
bash setup.sh
```

After starting the container you will see an output like

```
To access the notebook, open this file in a browser:
        file:///pathtojupyter/nbserver-15-open.html
    Or copy and paste one of these URLs:
        http://*********:8888/?token=*****************************************
     or http://127.0.0.1:8888/?token=*****************************************
```

You can then copy and paste one of the URLs into your browser to open the jupyter UI.

Once in jupyter you can click on *GettingStarted.ipynb* to get an interactive demo of the kit.
