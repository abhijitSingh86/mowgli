# Mowgli
[![CircleCI](https://circleci.com/gh/abhijitSingh86/mowgli.svg?style=svg)](https://circleci.com/gh/abhijitSingh86/mowgli)
> A man-trained boy would have been badly bruised, for the fall was a good fifteen feet, but Mowgli fell as Baloo had
> taught him to fall, and landed on his feet.
>
>  -- Rudyard Kipling, The Jungle Book

Mowgli is a _Python_ service for classifying intents.
You need a Python version >= 3.7. 

## Documentation
There is a openapi 3.0 documentation for the api at `doc/api.yaml`.

## Run tests
```bash
pipenv run pytest
```

## Run linter
```bash
pipenv run lint
```

## Run development server
```bash
pipenv run mowgli/run.py
```

    . Create an endpoint with GET /api/v1/intent
    . create a tensorflow network
    . train and test
    . export the weights..
    . use the weights to initialize the tensor flow for inference
    . write test cases to evaluate few scenarios
    

trainfing data


1,Text
1,Test is, Test ==> (1, (Test is,Test)) 



