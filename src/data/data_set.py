# -*- coding: utf-8 -*-
import numpy as np

class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool
        If this flag is set, then all labels which are not `targetDigit` will
        be transformed to False and `targetDigit` bill be transformed to True.
    targetDigit : string
        Label of the dataset, e.g. '7'.

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    oneHot : bool
    targetDigit : string
    """

    def __init__(self, data, oneHot=True, targetDigit='7'):

        # The label of the digits is always the first fields
        # Doing normalization
        self.input = (1.0 * data[:, 1:])/255
        self.label = data[:, 0]
        self.oneHot = oneHot
        self.targetDigit = targetDigit

        # Transform all labels which is not the targetDigit to False,
        # The label of targetDigit will be True,
        if oneHot:
            #print(self.label[0])
            oneHotEncoding = np.zeros((len(self.label), 10), dtype=np.uint8)
            for i in range(0, len(self.label)):
                oneHotEncoding[i, self.label[i]] = True
            self.label = oneHotEncoding
            #print(self.label[0])

    def __iter__(self):
        return self.input.__iter__()
