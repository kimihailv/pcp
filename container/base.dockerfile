FROM pytorch/pytorch:latest

RUN conda install -c conda-forge scikit-learn &&\
    conda install -c conda-forge k3d &&\
    conda install h5py &&\
    conda install -c pytorch faiss-cpu &&\
    pip install wandb matplotlib &&\
    echo 'alias jn="jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root --no-browser"' >> ~/.bashrc
