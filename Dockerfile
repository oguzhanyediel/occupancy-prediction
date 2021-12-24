FROM continuumio/miniconda3:4.9.2

COPY . /app

RUN conda install -c conda-forge pystan
RUN conda install -c conda-forge prophet
RUN conda install -c anaconda cython
RUN conda install -c anaconda numpy
RUN conda install -c anaconda pandas
RUN chmod 755 /app/main.py

ENTRYPOINT ["python3", "/app/main.py"]
