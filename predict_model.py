import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when

# Read input file path from command line argument
if len(sys.argv) != 2:
    print("Usage: spark-submit predict_model.py <input_dataset_path>")
    sys.exit(1)

input_path = sys.argv[1]

# Create Spark session
spark = SparkSession.builder.appName("MLPrediction").getOrCreate()

# Load input dataset from the provided path
data = spark.read.csv(input_path, header=True, inferSchema=True, sep=";")

# Clean column names
data = data.toDF(*[col.strip('"') for col in data.columns])

# Assemble feature vector
feature_cols = [col for col in data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembled_data = assembler.transform(data)

# Create a binary label (same as training)
binary_data = assembled_data.withColumn("binaryLabel", when(data["quality"] > 5, 1).otherwise(0))

# Load Logistic Regression model from S3
logr_model = LogisticRegressionModel.load("s3://kevin-project-2-files/logistic_model")

# Predict
logr_predictions = logr_model.transform(binary_data)

# Evaluate F1 Score
evaluator = MulticlassClassificationEvaluator(labelCol="binaryLabel", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(logr_predictions)

print(f"\n=== Logistic Regression Model ===")
print(f"F1 Score: {f1_score:.4f}")

# Stop Spark session
spark.stop()
