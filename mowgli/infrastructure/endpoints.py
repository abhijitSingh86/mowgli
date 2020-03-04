from flask import Flask, request

from mowgli.model import intent_classifier

APP = Flask(__name__)


@APP.route('/ping')
def ping():
    return 'PONG'


def is_valid(incoming_request):
    return incoming_request.is_json and 'message' in incoming_request.get_json()


@APP.route('/intent', methods=['POST'])
def classify_intent():
    if not is_valid(request):
        return "Message is not present", 400

    message = request.get_json()['message']
    intent, probability = intent_classifier.classify(message)
    return {'intent': {'name': intent, 'probability': probability}}
