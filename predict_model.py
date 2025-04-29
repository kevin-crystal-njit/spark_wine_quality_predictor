import sys
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when

if len(sys.argv) != 2:
    print("Usage: spark-submit predict_model.py <input_dataset_path>")
    sys.exit(1)

input_path = sys.argv[1]

spark = SparkSession.builder.appName("MLPrediction").getOrCreate()

logger = logging.getLogger('py4j')
logger.setLevel(logging.ERROR)
spark.sparkContext.setLogLevel("ERROR")

data = spark.read.csv(input_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip('"') for col in data.columns])

feature_cols = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(data)

binary_data = assembled_data.withColumn("binaryLabel", when(data["quality"] > 5, 1).otherwise(0))

logr_model = LogisticRegressionModel.load("s3://kevin-project-2-files/logistic_model")
logr_predictions = logr_model.transform(binary_data)

evaluator = MulticlassClassificationEvaluator(labelCol="binaryLabel", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(logr_predictions)

print(f"\n=== Logistic Regression Model ===")
print(f"F1 Score: {f1_score:.4f}")

spark.stop()

