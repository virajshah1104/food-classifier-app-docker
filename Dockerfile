FROM tensorflow/tensorflow
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 3000
EXPOSE $PORT
ENTRYPOINT [ "python" ] 
CMD [ "app.py" ] 