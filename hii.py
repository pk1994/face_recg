import zerorpc
from flask import jsonify

def main(dumped):
 client=zerorpc.Client();
 client.connect("tcp://127.0.0.1:4242")
 resp= client.classifyFile(dumped) 
 return resp;
 