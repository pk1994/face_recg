from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
import argparse
import sys
from datetime import datetime


img_path=os.getcwd()+'/test_images/test.png'
modeldir = './model/20180408-102900.pb'
classifier_filename = './class/classifier_3.pkl'
npy='./npy'
#train_img="./train_img"
def main(args):
    with tf.Graph().as_default():

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1), log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)#use only one npy file

            minsize = 20
            threshold = [0.6, 0.7, 0.7]
            factor = 0.709
            frame_interval = 3
            batch_size = args.batch_size
            image_size = args.image_size
            input_image_size = 160
            Names = ['Dr. Sai', 'aakash', 'abhilash', 'abhinav', 'aditya', 'ainesh', 'ainindita', 'ajit', 'akash', 'akhil', 'alex', 'amar', 'amit', 'ankit', 'ankush', 'anshuman_chand', 'anurag', 'aravindhan', 'arpita', 'ashish', 'bharat', 'chakradhar', 'chavi', 'chikkanna', 'chiranjeevi', 'deepa', 'devi_prasad', 'diana', 'dipesh', 'gautam', 'geetika', 'girish', 'guncha', 'hanaa', 'harsha', 'himanshu', 'ijis', 'iranna', 'jaicharan', 'kaushal', 'kiran', 'lokeshwar', 'manideep', 'manjushree', 'mayank', 'mohit', 'monika', 'mounika', 'mounika_g', 'mrinal', 'mukesh', 'mukul', 'neelmani', 'nethravathi', 'nitesh', 'nivedita', 'pallavi', 'payal', 'prabudh', 'prateek', 'prateek_agarwal', 'pratik', 'prayashi', 'preeti', 'priti', 'pujitha', 'purusoth', 'radhika', 'rahul_rai', 'rajesh', 'rakesh', 'raman', 'saad', 'sagorica', 'sai', 'sailesh', 'saloni', 'sarthak', 'sathish', 'saurabh', 'sharanjeet', 'shifu', 'shivam', 'shresth', 'shubhi', 'sinwan', 'siva', 'sivankar', 'somya', 'sourav', 'sreeparna', 'srikiran', 'srujay', 'subhodeep', 'subranshu', 'sudha', 'sumit', 'sumit_tandon', 'suvajit', 'tanu', 'tapobrata', 'tiasha', 'torsha', 'utkarsh', 'vaibhav', 'vaishnavi', 'vignesh', 'vijay', 'vishal_deshpande', 'yash', 'yatish', 'yogesh']
            #Names.sort()
            #print('Names: ',Names)

            facenet.load_model(modeldir)
            result_names=[]
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)

            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            t=datetime.now()
            c = 0
            prevTime = 0
            frame = cv2.imread(img_path,0)

            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is too close')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),                                          interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)


                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        t2=datetime.now()
                        for H_i in Names:
                            if Names[best_class_indices[0]] == H_i:
                                result_names.append({'person':Names[best_class_indices[0]],	'coords':bb[i]})
                    print("Prediction: ",result_names)



    return(result_names)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

   #parser.add_argument('data_dir', type=str,
   #help='Path to the data directory containing aligned LFW face patches.')
   # parser.add_argument('model', type=str,
   #help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
   #parser.add_argument('classifier_filename',
   #help='Classifier model file name as a pickle (.pkl) file. ' +
   #'For training this is the output and for classification this is an input.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    return parser.parse_args(argv)

if __name__ == '__main__':
    k=main(parse_arguments(sys.argv[1:]))
    print('asdfasfd', k)
