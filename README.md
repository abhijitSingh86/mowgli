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

Add your content i.e. the intent data into the existing nlu.md file. Use Lasagna tool to create new set of test,train 
and labels csv files. Once the dataset is created, use create_model.py(just uncomment the last line of function call and 
while committing comment it again) to create the network and evaluate the performance of the network. Which means more 
or less looking at the accuracy and validation accuracy and having multiple runs with varying params to see something 
changes. Once model have sufficient accuracy and also it passes all existing test cases fell free to commit and push the
 changes.
 
 And that's it! ...  



