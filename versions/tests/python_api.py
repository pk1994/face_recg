import flask
from flask import request, jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)


@app.route('/testget', methods=['GET'])
def get():
  return jsonify({'status': 'success'})
    
    
@app.route('/testpost', methods=['POST'])
def post():
  #print(request.form['image'])
  print(request.data)
  #print(request.args['image'])
  return jsonify({'status': 'success'})

app.run()
