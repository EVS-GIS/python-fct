#FROM ubuntu:focal
FROM python:3.8-buster as build
MAINTAINER rototo

RUN  export DEBIAN_FRONTEND=noninteractive
ENV  DEBIAN_FRONTEND noninteractive
RUN  dpkg-divert --local --rename --add /sbin/initctl

RUN apt-get update -qq && apt-get install -y \
  libssl-dev \
  libcurl4-gnutls-dev \
  libudunits2-dev \
  libgdal-dev \
  libsodium-dev

RUN apt-get update -qq && apt-get install -y \
  python3-dev \
  python3-rtree \
  python3-pip \
  libcairo2-dev

#RUN apt-get update -qq && apt-get install -y \
#  git-core

RUN echo "/opt/lib/python3.8/site-packages/" > /usr/local/python3.8/site-packages/opt.path \
 && git clone https://github.com/tramebleue/fct-cli /opt/fct-cli \
 && cd /opt/fct-cli \
 && python3 -m pip install --prefix /opt --upgrade pip \
 && python3 -m pip install --prefix /opt -r requirements.txt \
 && python3 -m pip install --prefix /opt sklearn \
 && python3 -m pip install --prefix /opt -r recommended.txt

RUN cd /opt/fct-cli \
 && git submodule init \
 && git submodule update --remote \
 && python3 -m pip install --prefix /opt .

FROM python:3.8-buster
MAINTAINER rototo

RUN apt-get update -qq && apt-get install -y \
  libssl1.1 \
  libcurl3-gnutls \
  libcurl4 \
  libudunits2-0 \
  libudunits2-data \
  libgdal20 \
  libsodium23

RUN apt-get update -qq && apt-get install -y \
  python3-rtree \
  libcairo2
  
RUN addgroup --gid 1000 fct \
 && adduser --uid 1000 --gid 1000 --disabled-password --gecos "" fct

COPY --from=build /opt /usr/local/
#USER fct