import requests
import sys
import base64

imgb64 = open(sys.argv[1], 'rb').read()
imgb64 = base64.b64encode(imgb64).decode()
url = 'https://food-classifier-dev.us-west-1.elasticbeanstalk.com:3000/predict'
headers = {
        'Content-type': "application/json"
    }
obj = {'image': imgb64}
url1 = 'https://food-classifier-dev.us-west-1.elasticbeanstalk.com'

response1 = requests.get(url1)
print(response1.text)

#response = requests.post(url, headers=headers, json=obj)
#print(response.text)