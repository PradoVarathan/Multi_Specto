FROM python:3.7

RUN apt-get -y update
RUN apt-get -y install git
RUN mkdir /Output
RUN mkdir /Input
WORKDIR /Input
RUN cd /Input
RUN git clone https://github.com/PradoVarathan/Multi_Specto
RUN cd Multi_Specto
RUN pip install -r requirements.txt

CMD ["python", "/Input/test_run.py"]

