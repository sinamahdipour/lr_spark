import sys
from pyspark import SQLContext
from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from texttable import Texttable
import os
#os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'


args = sys.argv
sc = SparkContext('local', 'test')
sqlContext = SQLContext(sc)

train_df = sqlContext.read.format("com.databricks.spark.csv") \
    .options(header='false', inferschema=True).load(args[1])

test_df = sqlContext.read.format("com.databricks.spark.csv") \
    .options(header='false', inferschema=True).load(args[2])


column_names = ['_c' + str(i) for i in range(14)]
assembler = VectorAssembler(inputCols=column_names[:13], outputCol="features")
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)
trainingData = train_df.select(['features', '_c13'])
testData = test_df.select(['features', '_c13'])

# Creating a linear regression model
lr = LinearRegression(maxIter=100, regParam=0, elasticNetParam=0, labelCol="_c13")
model = lr.fit(trainingData)


# print the coefficients table:
t = Texttable()
t.set_max_width(0)
headers = ['c_' + str(i) for i in range(1, 14)]
headers = ['Intercept'] + headers
vals = np.append(model.intercept, np.array(model.coefficients))
t.add_rows([headers, vals])
print(t.draw())

predictions = model.transform(testData)
# print('First 5 predictions...')
# predictions.select("prediction","_c13","features").show(5)


# draw the figure for predictions vs. ground truth
dt = predictions.toPandas()
plt.scatter(dt['prediction'], dt['_c13'])
plt.xlabel("Prediction")
plt.ylabel("Ground truth")
x1 = [0, np.maximum(np.max(dt['prediction']), np.max(dt['_c13']))]
y1 = [0, np.maximum(np.max(dt['prediction']), np.max(dt['_c13']))]
plt.plot(x1, y1)
plt.show()

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="_c13", metricName="rmse")
print("Root Mean Squared Error (RMSE) on test data = %g" % evaluator.evaluate(predictions))
