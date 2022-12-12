FROM euclia/jaqpotpy-inference
#FROM python:3.8
MAINTAINER Jason Sotiropoulos <jasonsoti1@gmail.com>
MAINTAINER Pantelis Karatzas <pantelispanka@gmail.com>

ADD src /jaqpot-inference/src
ADD application.py /jaqpot-inference/application.py
ADD requirements.txt /jaqpot-inference/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /jaqpot-inference/requirements.txt
#RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

# Expose the ports we're interested in
EXPOSE 8002

CMD ["python","/jaqpot-inference/application.py"]