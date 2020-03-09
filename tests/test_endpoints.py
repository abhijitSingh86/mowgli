from mowgli.infrastructure import endpoints

endpoints.APP.testing = True

with endpoints.APP.test_client() as client:
    def test_should_return_status_ok():
        response = client.get('/ping', content_type='html/text')

        assert 200 == response.status_code
        assert b'PONG' == response.data


    def test_should_return_classified_hello_intent(mocker):

        request_json = {
            'message': 'Hello'
        }
        response = client.post('/intent', json=request_json)
        actual = response.get_json()
        expected = {'intent': {'name': 'greet', 'probability': 1.0}}

        assert 200 == response.status_code
        assert expected == actual


    def test_should_return_classified_leave_intent(mocker):

        request_json = {
            'message': 'Show me my leave balance'
        }
        response = client.post('/intent', json=request_json)
        actual = response.get_json()
        expected = {'intent': {'name': 'leave_budget', 'probability': 1.0}}

        assert 200 == response.status_code
        assert expected == actual

        request_json = {
            'message': 'my leave balance'
        }
        response = client.post('/intent', json=request_json)
        actual = response.get_json()
        expected = {'intent': {'name': 'leave_budget', 'probability': 1.0}}

        assert 200 == response.status_code
        assert expected == actual


    def test_should_return_400():
        invalid_json = {}
        response = client.post('/intent', json=invalid_json)

        assert 400 == response.status_code


    def test_should_return_400_for_empty_body():
        response = client.post('/intent')

        assert 400 == response.status_code
