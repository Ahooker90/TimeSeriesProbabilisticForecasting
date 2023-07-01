from utils import util

stockDf = util.CreatePdDataframeForSingleStockPrice()
#util.PrintDF(stockDf,'Open')
trainData, testData, valData = util.CreateTrainTestValidationSet(stockDf)

util.PrintDF(trainData,"Open")
util.PrintDF(testData,"Open")
util.PrintDF(valData,"Open")









