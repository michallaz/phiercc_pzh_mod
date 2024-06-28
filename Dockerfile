FROM python:3.11.8-bookworm
LABEL maintainer "Michal Lazniewski <mlazniewski@pzh.gov.pl>"


ARG DEBIAN_FRONTEND=noninteractive
#continuumio/miniconda3

RUN apt update && apt upgrade -y
RUN apt install libopenblas-dev gfortran libboost-all-dev -y

# sciaganie scipy i podmienianie plikow z modulu cluster
WORKDIR /opt
RUN git clone https://github.com/scipy/scipy.git
WORKDIR /opt/scipy
COPY cluster/hierarchy.py /opt/scipy/scipy/cluster/hierarchy.py
COPY cluster/_hierarchy.pyx /opt/scipy/scipy/cluster/_hierarchy.pyx

RUN git submodule update --init
RUN python -m pip install -r requirements/all.txt
RUN python dev.py build
RUN cp -r build-install/lib/python3.11/site-packages/scipy/ /usr/local/lib/python3.11/

# instalowanie poprawnej wersji sharred-array
WORKDIR /opt
RUN git clone https://gitlab.com/moon548834/shared-array.git
WORKDIR /opt/shared-array
RUN python setup.py install
RUN pip install pandas numba Click
RUN pip install --no-deps pHierCC
COPY getDistance_github.py /usr/local/lib/python3.11/site-packages/pHierCC/getDistance.py
COPY pHierCC_github.py /usr/local/lib/python3.11/site-packages/pHierCC/pHierCC.py

WORKDIR /

