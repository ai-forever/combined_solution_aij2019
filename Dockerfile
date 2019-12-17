FROM ubuntu

RUN apt -y update
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN apt -y install build-essential libpcre3-dev python3 python3-pip curl zip unzip swig

RUN echo "ru_RU.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
ENV LANG ru_RU.UTF-8

RUN ln -s /usr/bin/python3 /usr/bin/python
COPY ./download_models.py /var/download_models.py
RUN python /var/download_models.py
RUN curl -sL https://github.com/dangerink/udpipe/archive/load_binary.zip -o /tmp/udpipe.zip &&     cd /tmp &&     unzip -qo /tmp/udpipe.zip  &&     cd /tmp/udpipe-load_binary/releases/pypi &&     ./gen.sh 1.2.0.1.0 &&     cd ufal.udpipe &&     python3 setup.py install &&     cd /tmp &&     rm -rf /tmp/udpipe*
RUN pip3 install numpy==1.17.2 scipy sklearn pandas==0.24.2 attrs==19.1.0 lightgbm==2.2.3 nltk==3.2.5 gensim==3.8.0 torch transformers==2.1.1 catboost pytorch_pretrained_bert==0.6.2 matplotlib==3.0.3 python-Levenshtein sklearn_crfsuite fastai keras tqdm pymorphy2 summa pymystem3 pymorphy2 pymorphy2-dicts-ru jellyfish flask requests tensorflow==1.14.0 sentencepiece==0.1.83 tf-sentencepiece==0.1.83 tensorflow-hub==0.6.0 razdel==0.4.0
RUN python -c "import pymystem3.mystem ; pymystem3.mystem.autoinstall()"
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt');"
