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
#import classifier_new
import hii
from PIL import Image
from binascii import a2b_base64
from flask_cors import CORS
import time
import numpy as np
import zerorpc
import codecs, json
from preprocess import preprocesses
from numpy import array
from io import BytesIO
import base64
import io
import pickle
from scipy import misc

app = Flask(__name__)
CORS(app)
## Logs are logged into the server.log file.
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))

def stringToRGB(img_str):
    tempBuff = BytesIO()
    tempBuff.write(img_str)
    tempBuff.seek(0) #need to jump back to the beginning before handing it off to PIL
    Image.open(tempBuff)
    print('image size',len(image.split()))
    # image = Image.merge("RGB", (r, g, b))
    image.save("test_images/test.png")
    # return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	
	
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    dataBytesIO = io.BytesIO(imgdata)
    return Image.open(dataBytesIO)

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    #r,g,b,_ = cv2.split(np.array(image))
    #print('after spliting image',)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)
    #return cv2.merge((r,g,b))
	
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
       
        startTime = time.time()
		# convert string of image data to uint8
        # nparr = np.fromstring(request.data.decode("utf-8").split(",", 1)[1], np.uint8)
        # print('nparr is',nparr)
		# # decode image
        # img_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print('image data',img_data)
        encoded = request.data.decode("utf-8").split(",", 1)[1]
        #img_data = a2b_base64(encoded)
        # print('size of rgb',rgb_image.size)
        #img = img_data.convert('RGB')
        # with open("test_images/test.png", "wb") as f:
          # f.write(img_data)
        
        #rgba_image = Image.open("test_images/test_32.png")
        #print('size of rgba',rgba_image.size)
        #image = array(rgba_image).reshape(1,540,404,3)
        # rgb_image = rgba_image.convert('RGB')
        print('type of data',type(encoded))
        rgba_image = stringToImage(encoded)
        #print('rgba_image size',rgba_image.size)
        #rgba_image.save("test_images/rgba.png")
        # rgb_image = rgba_image.convert('RGB')
        rgb_image = toRGB(rgba_image)
        #misc.imsave(os.getcwd()+'/test_images/before_dump.png',rgb_image)
        print('rgb image type',type(rgb_image))
        print('rgb image type',rgb_image.shape)
        print('rgb image dtype',rgb_image.dtype)
        # rgb_image.save("test_images/test.png")
        #print('size of rgb',rgb_image.size)
		
        #print('Image Saved')
        print('timetaken to save image::::',time.time()-startTime)
        #facecrop("./test_images/test1.png")

		
        # input_datadir = './test_images'
        # output_datadir = './pre_img'
        # obj=preprocesses(input_datadir,output_datadir)
        # nrof_images_total,nrof_successfully_aligned=obj.collect_data()
		
        # print('Total number of images: %d' % nrof_images_total)
        # print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
		
        pickled_dumped = pickle.dumps(rgb_image, protocol=2)
        resp=hii.main(pickled_dumped)
        
        
        #resp =classifier_new. RPCCom().classifyFile('./test_images/test.png')
		
                          
		#client.invoke("classifyFile", JPG_FILE, function(error, res, more) 
        #classifier_new.main()
        #rpccall();
        
        print(' total time taken:::', time.time() - startTime);
        # print("datatype of response:", type(resp))
        # print('length', len(resp))
        # for i in range(len(resp)):
            # cords = resp[i]['coords']
            # resp[i]['coords'] = [];
            # for cord in cords:
                # resp[i]['coords'].append( int(cord) )
        # print("Parsed resp:", resp)
        elements = resp.split(',')
        resp = [{'person':elements[0],'coords':elements[-4:]}]
        print('responseeeeee::::::',resp)

        return jsonify(resp);

    else:
    	return "Where is the image?"
		
		
# def facecrop(image):
    # facedata = "haarcascade_frontalface_alt.xml"
    # cascade = cv2.CascadeClassifier(facedata)

    # img = cv2.imread(image)

    # minisize = (img.shape[1],img.shape[0])
    # miniframe = cv2.resize(img, minisize)

    # faces = cascade.detectMultiScale(miniframe)

    # for f in faces:
        # x, y, w, h = [ v for v in f ]
        # cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        # sub_face = img[y:y+h, x:x+w]
        # fname, ext = os.path.splitext(image)
        # cv2.imwrite('./test_images/test.png', sub_face)
        # print('image cropped')



    # return

@app.route('/testget', methods = ['GET'])
def test_get():
  return "Tested Get."
  
  
def rpccall():

    client=zerorpc.Client();
    client.connect("tcp://127.0.0.1:4242")
    resp= client.classifyFile('./test_images/test.png') 
    return resp;


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
