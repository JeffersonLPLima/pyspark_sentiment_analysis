import os
sentiment140_csv = './data/training.1600000.processed.noemoticon.csv'
if(not os.path.exists(sentiment140_csv)):
    print('Sentiment data not found! Download sentiment data at https://www.kaggle.com/kazanova/sentiment140')
    exit()
     
import findspark
findspark.init()
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

spark = SparkSession\
    .builder\
    .master('local[*]')\
    .appName("SentimentAnalysis")\
    .getOrCreate()

df = SQLContext(spark).read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load(sentiment140_csv) #to download https://www.kaggle.com/kazanova/sentiment140
df = df[['_c0', '_c5']].selectExpr("_c5 as text", "_c0 as target")
df.dropna()
df.printSchema()
print(df.show(5))

(train_set, val_set, test_set) = df.randomSplit([0.7, 0.1, 0.2], seed = 2000)
print(f"{train_set.count()} train samples")
print(f"{val_set.count()}   val samples")
print(f"{test_set.count()}  test samples")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "target", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
val_df = pipelineFit.transform(val_set)
train_df.show(5)



lr = LogisticRegression(maxIter=100)
lrModel = lr.fit(train_df)
predictions = lrModel.transform(val_df)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator.evaluate(predictions)


print(f'Current Accuray:{predictions.filter(predictions.label == predictions.prediction).count() / float(val_set.count())}')

 
print("Using model...")
sentence = ""
while sentence.lower()!="q":
    sentence = input('Insert sentence (q to exit):')
    rdd1 = spark.sparkContext.parallelize([sentence])
    row_rdd = rdd1.map(lambda x: Row(x))
    test_df = SQLContext(spark).createDataFrame(row_rdd,['text'])
    preds = lrModel.transform(pipelineFit.transform(test_df))
    sent = "POSITIVE" if preds.select("prediction").rdd.flatMap(lambda x: x).collect()[0] == 0.0 else "NEGATIVE"
    print(sent)
spark.stop()


