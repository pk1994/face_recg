"""
@author  : Rakesh Roshan
@contact : rakesh.roshan@affineanalytics.com

A set of functions to start the API serverand listen to the requests.
Returns the JSON output from running models and IOU their outputs.
"""
import cv2
import json
from flask import Flask, url_for, send_from_directory, request, jsonify
import logging, os
from werkzeug import secure_filename
import classifier_new

from base64 import b64decode

from PIL import Image

def blend_value(under, over, a):
    return (over*a + under*(255-a)) / 255

def blend_rgba(under, over):
    return tuple([blend_value(under[i], over[i], over[3]) for i in (0,1,2)] + [255])

white = (255, 255, 255, 255)



app = Flask(__name__)
## Logs are logged into the server.log file.
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))


@app.route('/testpost', methods = ['POST'])
def api_root():
    """
    Reads the image from request.
    Runs the model.
    Returns the JSON output.
    Args: HTTP post request with an image file specified to the key 'image'
    """
    app.logger.info('Project_Home:' + PROJECT_HOME)
    if request.method == 'POST':
        #print(request.form['image'])
        ## Read the image file.
        #img = request.form['image']
        #print(request.form['image'].encode('utf-8'))
        #img = base64.b64decode(request.form['image'].encode('utf-8')).decode('utf-8')
        #img = request.form['image'].decode('base64')
        #imgData = request.form['image'][:-1]
        #imgData = imgData[22:] 
        #print(imgData)
        #img = base64.b64decode(imgData)
        encoded = request.form['image'].split(",", 1)[1] 
        print(encoded)
        img_data = b64decode(encoded)
        with open("test_images/test_32.png", "wb") as f:
          f.write(img_data)
        rgba_image = Image.open("test_images/test_32.png")
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save("test_images/test.png")
        response = classifier_new.main(classifier_new.parse_arguments([])) 
        print(response)
        ## Return the JSON response to the API request.
        return jsonify(response) 
    else:
    	return "Where is the image?"

@app.route('/testget', methods = ['GET'])
def test_get():
  return "Tested Get."


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
