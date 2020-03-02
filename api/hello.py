
from flask import Flask, request

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello, Mowgli!'

@app.route('/intent', methods = ['POST'])
def getIntent():
    msg = request.args.get('msg')
    return msg

if __name__ == '__main__':
    app.run()
