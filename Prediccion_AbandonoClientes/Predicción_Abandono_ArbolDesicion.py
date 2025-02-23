# Databricks notebook source
#Leendo el Dataset
dataset = spark.read.format("parquet").load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/dataset_EstadoClientes")
dataset.show()

# COMMAND ----------

#Creando el modelo de Arbol de Desicion
from pyspark.ml.classification  import DecisionTreeClassifier

# COMMAND ----------

algoritmo = DecisionTreeClassifier(
    labelCol="Estado_del_Cliente_indexado",
    featuresCol="features"
)

# COMMAND ----------

#Configurando el evaluador
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

evaluador = MulticlassClassificationEvaluator(
    labelCol="Estado_del_Cliente_indexado",
    predictionCol="prediction",
    metricName="accuracy"
)

# COMMAND ----------

#Utlitario para definir los valores los valores que se evaluaran en los parametros de mi algoritmo
from pyspark.ml.tuning import ParamGridBuilder

# COMMAND ----------

# MAGIC %md
# MAGIC **Malla de Parametros a calibrar**

# COMMAND ----------

mallaParametros = ParamGridBuilder().\
addGrid(algoritmo.maxDepth,[5, 8, 10]).\
addGrid(algoritmo.impurity,["entropy", "gini"]).\
build()

# COMMAND ----------

# MAGIC %md
# MAGIC **Validacion Cruzada**

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator

# COMMAND ----------

validacionCruzada = CrossValidator(
    estimator=algoritmo,
    estimatorParamMaps=mallaParametros,
    evaluator=evaluador,
    numFolds=5
)

# COMMAND ----------

#Ejecucion de la validacion cruzada para obtener el mejor modelo
modeloGenerados = validacionCruzada.fit(dataset)

# COMMAND ----------

#Extrayendo el mejor modelo
modelo=modeloGenerados.bestModel

# COMMAND ----------

# MAGIC %md
# MAGIC **Validacion del Modelo**

# COMMAND ----------

dfPrediccion = modelo.transform(dataset)
dfPrediccion.show()

# COMMAND ----------

dfPrediccion.select(
  dfPrediccion["Estado_del_Cliente_indexado"],
  dfPrediccion["prediction"],
  dfPrediccion["probability"]
).show(20, False)

# COMMAND ----------

#Configuramos el evaluador
evaluador = MulticlassClassificationEvaluator(
  labelCol="Estado_del_Cliente_indexado", 
  predictionCol="prediction", 
  metricName="accuracy"
)
 
#Evaluamos las predicciones
evaluador.evaluate(dfPrediccion)

# COMMAND ----------

#Almacenamos el modelo
modelo.write().overwrite().save("dbfs:/FileStore/_AbandonoClientesAnalisis/output/modelo_arbol_de_decision/")
