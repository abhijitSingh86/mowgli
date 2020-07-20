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
    
## Steps to train the neural network with more data

Update the test and train files under resources. Lasagna can be used to transform nlu.md format to simple CSV.
To train 

`pipenv run train`

This updates the model and tokenizer instance.
 



