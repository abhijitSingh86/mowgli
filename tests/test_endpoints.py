from mowgli.infrastructure import endpoints

endpoints.app.testing = True

with endpoints.app.test_client() as client:
    def test_should_return_status_ok():
        response = client.get('/ping', content_type='html/text')

        assert 200 == response.status_code
        assert b'PONG' == response.data


    def test_should_return_classified_intent(mocker):
        predict_mock = mocker.patch('mowgli.model.intent_classifier.classify')
        predict_mock.return_value = ('foo_intent', 1.0)
        request_json = {
            'message': 'Hey'
        }
        response = client.post('/intent', json=request_json)
        actual = response.get_json()
        expected = {'intent': {'name': 'foo_intent', 'probability': 1.0}}

        assert 200 == response.status_code
        assert expected == actual
        predict_mock.assert_called_with(request_json['message'])


    def test_should_return_400():
        invalid_json = {}
        response = client.post('/intent', json=invalid_json)

        assert 400 == response.status_code


    def test_should_return_400_for_empty_body():
        response = client.post('/intent')

        assert 400 == response.status_code
