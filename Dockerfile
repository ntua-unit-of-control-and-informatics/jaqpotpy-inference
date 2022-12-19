FROM euclia/jaqpotpy-inference
MAINTAINER Jason Sotiropoulos <jasonsoti1@gmail.com>
MAINTAINER Pantelis Karatzas <pantelispanka@gmail.com>

ADD src /jaqpot-inference/src
ADD app /jaqpot-inference/app
ADD requirements.txt /jaqpot-inference/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /jaqpot-inference/requirements.txt

EXPOSE 8002

CMD ["uvicorn", "jaqpot-inference.app.application:app", "--host", "0.0.0.0", "--port", "8002"]
