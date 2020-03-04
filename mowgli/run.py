#!/usr/bin/env python3
import os

from mowgli.infrastructure import endpoints


def run():
    port = os.environ.get('PORT', 8080)
    endpoints.APP.run(port=port, debug=True, host='0.0.0.0')


if __name__ == '__main__':
    run()
