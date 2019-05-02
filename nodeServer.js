    
var zerorpc = require("zerorpc");
var startTime = new Date().getTime();

// Change this to a JPG to classify on your server.
var JPG_FILE = './test_images/test.png';

// Connect to Python RPC server.
var client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:4242");

// Invoke remote function call to classify a jpg file of our choice.
client.invoke("classifyFile", JPG_FILE, function(error, res, more) {
  // Print out classification.
  var endTime = new Date().getTime();
  console.log(res.toString('utf8'));
  console.log('totalTime :::' ,(endTime - startTime) / 1000)
 
});