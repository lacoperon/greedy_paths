'''
Function which allows for iterative changes to a cumulative variable
(ie tracking Average or Standard Deviation, or something more fancy like
a histogram binning results)

Input:
    dataPoint (one instance of data, to be added to aggData in some way)
    aggData (the object containing the aggregated data)

'''

def addToHistogram(hist_list, bin_width, dataPoint):


# TODO: find way to pass partial functions as arguments to this
from functools import partial

def add_and_aggregate(dataPoint, aggData, dataType="average"):
    assert dataType in ["average","variance","histogram"]
    if dataType == "average":
        curr_average, N = aggData
        updated_average = (curr_average * (float(N-1)/N)) + float(dataPoint) / N
        N += 1
        return [updated_average, N]
    if dataType == "variance":
        pass
