FROM openjdk:17-slim

RUN apt-get update && \
    apt-get install -y python3.9 python3-pip wget && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    pip install numpy

ENV SPARK_VERSION=3.5.4
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:/opt/spark/bin

RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

WORKDIR /app

COPY predict_model.py /app/

# ENTRYPOINT forces S3 argument and always runs predict_model.py
ENTRYPOINT ["bash", "-c", "if [ -z \"$1\" ]; then echo 'ERROR: You must provide an S3 path as argument.'; exit 1; fi; exec spark-submit /app/predict_model.py \"$@\""]
CMD []
