# Databricks notebook source
# MAGIC %md
# MAGIC Reference: https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/feature/BucketedRandomProjectionLSHModel.html

# COMMAND ----------

dbutils.fs.put('dbfs:/tmp/file1.csv', """text
Wall Decals Lamp Shades Armchairs Bed Sheets Night Lights Necklaces Decorative Pillow Covers Table Lamps Decorative Boxes Lamps Slumber Bags Figurines Tableware Plates Decorative Pillows Fancy-Dress Costumes Curtains Canvas Art Prints
iphone charger phone Gift case iPhone holder selfie-stick
""")

# COMMAND ----------

dbutils.fs.put('dbfs:/tmp/file2.csv', """text
Curtains & Valances Wall Decals & Stickers Beds Area Rugs Bedding Sets Activity Tables Lamps Doll Playsets Interlocking Block Building Sets Night Lights Armchairs & Accent Chairs Organizing Racks Table Lamps Desks Bed Sheets Bookcases
iphone case Apple ipod
""")

# COMMAND ----------

dbutils.fs.ls('/tmp')

# COMMAND ----------

import sparknlp
import json
import os
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
from pyspark.sql.functions import from_unixtime
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import *
from pyspark.sql.functions import explode, col
from pyspark.sql.functions import from_unixtime, to_date, asc, year, udf, explode, split, col, desc, length, rank, dense_rank, avg, sum
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, to_timestamp,date_format
from pyspark import StorageLevel
import pyspark.sql.functions as F
from sparknlp.pretrained import PretrainedPipeline
from collections import Counter
from sparknlp.base import Finisher, DocumentAssembler
from sparknlp.annotator import (Tokenizer, Normalizer,LemmatizerModel, StopWordsCleaner)
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.ml.feature import Normalizer, SQLTransformer
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.sql.functions import monotonically_increasing_id

import pandas as pd

# COMMAND ----------

primaryCorpus = spark.read.option("header","true").csv("/tmp/file1.csv")
secondaryCorpus = spark.read.option("header","true").csv("/tmp/file2.csv")

# COMMAND ----------

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

sentence = SentenceDetector()\
      .setInputCols("document")\
      .setOutputCol("sentence")\
      .setExplodeSentences(False)

tokenizer = Tokenizer()\
    .setInputCols(['sentence'])\
    .setOutputCol('token')

bertEmbeddings = BertEmbeddings\
 .pretrained('bert_base_cased', 'en') \
 .setInputCols(["sentence",'token'])\
 .setOutputCol("bert")\
 .setCaseSensitive(False)

embeddingsSentence = SentenceEmbeddings() \
      .setInputCols(["sentence", "bert"]) \
      .setOutputCol("sentence_embeddings") \
      .setPoolingStrategy("AVERAGE")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings","bert"]) \
    .setOutputCols("sentence_embeddings_vectors", "bert_vectors") \
    .setOutputAsVector(True)\
    .setCleanAnnotations(False)


explodeVectors = SQLTransformer() \
.setStatement("SELECT EXPLODE(sentence_embeddings_vectors) AS features, * FROM __THIS__")

vectorNormalizer = Normalizer() \
    .setInputCol("features") \
    .setOutputCol("normFeatures") \
    .setP(1.0)

similartyChecker = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=6.0,numHashTables=6)

# COMMAND ----------

pipeline = RecursivePipeline() \
      .setStages([documentAssembler,
        sentence,
        tokenizer,
        bertEmbeddings,
        embeddingsSentence,
        embeddingsFinisher,
        explodeVectors,
        vectorNormalizer,
        similartyChecker])

# COMMAND ----------

pipelineModel = pipeline.fit(primaryCorpus)
primaryDF = pipelineModel.transform(primaryCorpus)
secondaryDF = pipelineModel.transform(secondaryCorpus)

# COMMAND ----------

dfA = primaryDF.select("text","features","normFeatures").withColumn("lookupKey", md5("text")).withColumn("id",monotonically_increasing_id())
dfA.show()

# COMMAND ----------

dfB = secondaryDF.select("text","features","normFeatures").withColumn("id",monotonically_increasing_id())
dfB.show()

# COMMAND ----------

pipelineModel.stages[8].approxSimilarityJoin(dfA, dfB, 100, distCol="distance")\
     .select(col("datasetA.text").alias("idA"), \
            col("datasetB.text").alias("idB"), \
            col("distance")).show()

# COMMAND ----------

from pyspark.sql.functions import PandasUDFType, pandas_udf
import pyspark.sql.functions as F

dfA = dfA.withColumnRenamed('text','primaryText').withColumnRenamed('features', 'primaryFeatures')

dfB = dfB.withColumnRenamed('text','secondaryText').withColumnRenamed('features', 'secondaryFeatures')

joinedDF = dfA.join(dfB, "id", "inner").drop("id","normFeatures")

joinedDF.show()

# COMMAND ----------

from scipy.spatial.distance import cosine

finalDF = joinedDF.toPandas()

finalDF['cosine'] = finalDF.apply(lambda row: 1-cosine(row['primaryFeatures'], row['secondaryFeatures']), axis=1)
finalDF

# COMMAND ----------

