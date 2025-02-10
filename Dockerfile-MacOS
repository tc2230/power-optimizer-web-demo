FROM python:3.11.8-slim

RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install plotly
RUN pip install streamlit
RUN pip install mip

RUN apt-get update && apt-get install -y nano wget git build-essential pkgconf
#RUN apt update && apt upgrade
#RUN apt install wget

RUN mkdir /code
COPY . /code

RUN mkdir CBC
WORKDIR ./CBC

RUN wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
RUN chmod u+x coinbrew
RUN ./coinbrew fetch Cbc@master
RUN ./coinbrew build Cbc 
RUN cp /CBC/dist/lib/libCbc.so /usr/local/lib/python3.11/site-packages/mip/libraries/cbc-c-linux-x86-64.so

WORKDIR /code

CMD ["/bin/bash", "-c", "cd /code; streamlit run /code/web_example.py", "--server.fileWatcherType", "none"]
#CMD ["/bin/bash"]