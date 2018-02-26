'''
Function which allows for iterative changes to a cumulative variable
(ie tracking Average or Standard Deviation, or something more fancy like
a histogram binning results)

Input:
    dataPoint (one instance of data, to be added to aggData in some way)
    aggData (the object containing the aggregated data)

'''

class aggregateFunctions:
    '''
    TODO: Fill this in
    '''
    @staticmethod
    def average(dataPoint, aggData):
        if len(aggData) is not 2:
            x = len(aggData)
            raise Exception("Need 2 args to average, given {}".format(x))
        curr_average, N = aggData
        if N is 0:

        updated_average = (curr_average * (float(N-1)/N)) + float(dataPoint) / N
        N += 1
        return [updated_average, N]

    '''
    TODO: Fill this in
    '''
    @staticmethod
    def variance(dataPoint, aggData):
        if len(aggData) is not 3:
            x = len(aggData)
            raise Exception("Need 3 args to calculate Var, given {}".format(x))
        var, curr_avg, curr_squared_avg, N = aggData
        updated_avg = (curr_avg * (float(N-1)/N)) + float(dataPoint) / N
        upd_sqr_avg = (curr_squared_avg *(float(N-1)/N)) + float(dataPoint**2)/N
        N += 1

        var = upd_sqr_avg - (updated_avg ** 2)
        return [var, updated_avg, upd_sqr_avg, N]

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
        curr_variance, N = aggData
        updated_average =
