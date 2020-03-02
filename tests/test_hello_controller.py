import unittest

from flaskr.app import app

class MowgliTestCase(unittest.TestCase):

    def test_hello_world(self):
        """Test API response (GET request)"""
        client = app.test_client(self)
        response = client.get('/hello', content_type='html/text')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'Hello, Mowgli!')

if __name__ == "__main__":
    unittest.main()