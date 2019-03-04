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
from PIL import Image
from binascii import a2b_base64
from flask_cors import CORS
import time


app = Flask(__name__)
CORS(app)
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
        #print(request.data)
        encoded = request.data.decode("utf-8").split(",", 1)[1]
        img_data = a2b_base64(encoded)
        with open("test_images/test_32.png", "wb") as f:
          f.write(img_data)
        rgba_image = Image.open("test_images/test_32.png")
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save("test_images/test.png")
        startTime = time.time()
        resp = classifier_new.main(classifier_new.parse_arguments([]))
        resp['model_proc_time'] = time.time() - startTime;
        print(resp)
        return jsonify(resp)
    else:
    	return "Where is the image?"

@app.route('/testget', methods = ['GET'])
def test_get():
  return "Tested Get."


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
