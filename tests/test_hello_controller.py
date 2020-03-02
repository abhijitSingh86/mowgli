import unittest

from api.hello import app

class MowgliTestCase(unittest.TestCase):

    def test_hello_world(self):
        """Test API response (GET request)"""
        client = app.test_client(self)
        response = client.get('/hello', content_type='html/text')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'Hello, Mowgli!')

    def testGetIntentApi(self):
        """ test the intent api """
        client = app.test_client(self)
        msg = "Hello Balu"
        response = client.post('/intent?msg=' + msg, content_type='html/text')
        self.assertEqual(msg, response.data.decode("utf-8"))

        blankMsg = ""
        response = client.post('/intent?msg=' + blankMsg, content_type='html/text')
        self.assertEqual(blankMsg, response.data.decode("utf-8"))


        validMsg = "Get me my leaves"
        validResponse = "leave_budget"
        response = client.post('/intent?msg=' + validMsg, content_type='html/text')
        self.assertEqual(validResponse, response.data.decode("utf-8"))

if __name__ == "__main__":
    unittest.main()