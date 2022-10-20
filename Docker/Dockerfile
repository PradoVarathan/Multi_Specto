FROM python:3.6.13

RUN apt-get update -y && apt-get -y install apt-file && apt-get install -y python3-dev build-essential
RUN apt-get -y install git

RUN mkdir /Output
RUN mkdir /Input
WORKDIR /Input

RUN cd /Input
RUN git clone https://github.com/PradoVarathan/Multi_Specto
RUN git clone https://github.com/FunctionLab/ExPecto &&\
    cd ExPecto &&\
    sh download_resources.sh; tar xf resources_20190807.tar.gz &&\
    mv /Input/ExPecto/resources_20190807/* /Input/Multi_Specto/
RUN mv /Input/ExPecto/sources/deepsea_beluga_2002_features.tsv /Input/Multi_Specto/

RUN cd Input/Multi_Specto/
RUN python -m pip install --upgrade pip
RUN pip install biopython &&\
    pip install pandas &&\
    pip install numpy &&\
    pip install argparse &&\
    pip install tqdm &&\
    pip install h5py &&\
    pip install seaborn &&\
    pip install matplotlib &&\
    pip install scipy 
    
RUN pip install torch --no-cache-dir

CMD ["python", "/Input/Multi_Specto/test_run.py"]

