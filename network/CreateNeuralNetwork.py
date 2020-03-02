from tensorflow import keras
import pandas as pd
from Sanitizer import normalize

TRAIN_DATA_BASEPATH = "../data"

def loadData(fileName, isNormalizee = False):

    """

    data  = [
            9,data is Dirty#TOD
                ]
    """
    data = pd.read_csv(TRAIN_DATA_BASEPATH + "/"+fileName, sep='&&&', header=None)

    def customSplitter(text):
        i = text.index(',')
        return (text[0:i], text[i + 1:])

    """
        ex = [
            ['label', 'msg'   ]
            [ 9     , 'data is Dirty#TOD' ]
        ]
    """
    ex = pd.DataFrame(data.applymap(lambda x: customSplitter(x))[0].values.tolist(), columns=['label', 'msg'])
    normalizedData = normalize(ex)
    return ex

def begin():
    # Load train and label data
    # normalize oder lammitize oder lancasterStemmer
    # create network
    # feed
    print("beginning")
    traindData = loadData("train.csv", True)
    labelData  = loadData("labels.csv")

    #print(traindData)
    #print(labelData)


if __name__ == "__main__" :
    begin()