# Databricks notebook source
#Leendo el Dataset
dataset = spark.read.format("parquet").load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/dataset_EstadoClientes")
dataset.show()

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

#Configuramos el algoritmo
algoritmo = RandomForestClassifier(
  labelCol = "Estado_del_Cliente_indexado", 
  featuresCol = "features"
)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

evaluador = MulticlassClassificationEvaluator(
  labelCol = "Estado_del_Cliente_indexado",
  predictionCol = "prediction",
  metricName="accuracy"
)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

# COMMAND ----------

mallaDeParametros = ParamGridBuilder().\
addGrid(algoritmo.numTrees, [5, 8, 12]).\
addGrid(algoritmo.maxDepth, [5, 8, 10]).\
addGrid(algoritmo.impurity, ["entropy", "gini"]).\
build()

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

# COMMAND ----------

#Configuración de la validación cruzada
validacionCruzada = CrossValidator(
  estimator = algoritmo,
  estimatorParamMaps = mallaDeParametros,
  evaluator = evaluador,
  numFolds = 5 
)  

# COMMAND ----------

modelosGenerados = validacionCruzada.fit(dataset)

# COMMAND ----------

#Extracción del mejor modelo
modelo = modelosGenerados.bestModel

# COMMAND ----------

dfPrediccion = modelo.transform(dataset)
dfPrediccion.show()

# COMMAND ----------

dfPrediccion.select(
  dfPrediccion["Estado_del_Cliente_indexado"],
  dfPrediccion["prediction"],
  dfPrediccion["probability"], 
  dfPrediccion["rawPrediction"]
).show(20, False)

# COMMAND ----------

evaluador = MulticlassClassificationEvaluator(
  labelCol="Estado_del_Cliente_indexado", 
  predictionCol="prediction", 
  metricName="accuracy"
)
evaluador.evaluate(dfPrediccion)

# COMMAND ----------

#Almacenamos el modelo
modelo.write().overwrite().save("dbfs:/FileStore/_AbandonoClientesAnalisis/output/modelo_RandomForest/")
