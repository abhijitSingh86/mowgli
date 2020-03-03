from flask import Flask, request

from mowgli.model import intent_classifier

app = Flask(__name__)


@app.route('/ping')
def ping():
    return 'PONG'


def is_valid(request):
    return request.is_json and 'message' in request.get_json()


@app.route('/intent', methods=['POST'])
def classify_intent():
    if not is_valid(request):
        return "Message is not present", 400

    message = request.get_json()['message']
    intent, probability = intent_classifier.classify(message)
    return {'intent': {'name': intent, 'probability': probability}}
