FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install --upgrade pip
EXPOSE $PORT
ENTRYPOINT [ "python" ] 
CMD [ "app.py" ] 