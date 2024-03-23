FROM auchida/pandoc
RUN apt-get update\
 && apt-get install --no-install-recommends -y python3 python3-pip gcc python3-dev
RUN pip install pandoc-fignos --user
ENV PATH="${PATH}:/root/.local/bin"
WORKDIR /opt/docs
CMD ["/bin/bash"]
