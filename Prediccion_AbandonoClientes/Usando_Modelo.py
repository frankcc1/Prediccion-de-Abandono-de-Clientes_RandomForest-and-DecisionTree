# Databricks notebook source
from pyspark.sql.types import StructField,StructType
from pyspark.sql.types import StringType,DoubleType
import pyspark.pandas as pd
pd.set_option('display.max_rows',20)

# COMMAND ----------

dfRaw = spark.read.format("csv").option("header", "true").option("delimiter", ",").option("encoding", "ISO-8859-1").schema(
    StructType(
        [
            StructField("ID_Cliente",StringType(),True),
            StructField("Género",StringType(),True),
            StructField("Edad",DoubleType(),True),  
            StructField("Casado",StringType(),True),
            StructField("Estado",StringType(),True),
            StructField("Número_de_Referencias",DoubleType(),True),
            StructField("Antigüedad_en_Meses",DoubleType(),True),
            StructField("Oferta_de_Valor",StringType(),True),
            StructField("Servicio_Teléfonico",StringType(),True),
            StructField("Líneas_Múltiples",StringType(),True),
            StructField("Servicio_de_Internet",StringType(),True),
            StructField("Tipo_de_Internet",StringType(),True),
            StructField("Seguridad_en_Línea",StringType(),True),
            StructField("Respaldo_en_Línea",StringType(),True),
            StructField("Plan_de_Protección_de_Dispositivos",StringType(),True),
            StructField("Soporte_Premium",StringType(),True),
            StructField("Streaming_TV",StringType(),True),
            StructField("Streaming_Películas",StringType(),True),
            StructField("Streaming_Música",StringType(),True),
            StructField("Datos_Ilimitados",StringType(),True),
            StructField("Contrato",StringType(),True),
            StructField("Facturación_Sin_Papel",StringType(),True),
            StructField("Método_de_Pago",StringType(),True),
            StructField("Cargo_Mensual",DoubleType(),True), 
            StructField("Cargos_Totales",DoubleType(),True), 
            StructField("Total_de_Reembolsos",DoubleType(),True), 
            StructField("Total_de_Cargos_Adicionales_por_Datos",DoubleType(),True),  
            StructField("Total_de_Cargos_por_Larga_Distancia",DoubleType(),True),  
            StructField("Ingresos_Totales",DoubleType(),True),      
            StructField("Estado_del_Cliente",StringType(),True),
            StructField("Categoría_de_Cancelación",StringType(),True),
            StructField("Razón_de_Cancelación",StringType(),True)
        ]
    )
).load("dbfs:/FileStore/_AbandonoClientesAnalisis/Nuevos_Clientes.csv")

dfRaw.show()

# COMMAND ----------

dfpRaw = pd.from_pandas(dfRaw.toPandas())
dfpRaw

# COMMAND ----------

dfpRaw.isnull().sum()

# COMMAND ----------

dfpRaw["ID_Cliente"].count()

# COMMAND ----------

dfpRaw['Líneas_Múltiples'] = dfpRaw['Líneas_Múltiples'].fillna('No')
dfpRaw['Tipo_de_Internet'] = dfpRaw['Tipo_de_Internet'].fillna('Otro')
dfpRaw['Seguridad_en_Línea'] = dfpRaw['Seguridad_en_Línea'].fillna('No')
dfpRaw['Respaldo_en_Línea'] = dfpRaw['Respaldo_en_Línea'].fillna('No')
dfpRaw['Plan_de_Protección_de_Dispositivos'] = dfpRaw['Plan_de_Protección_de_Dispositivos'].fillna('No')
dfpRaw['Soporte_Premium'] = dfpRaw['Soporte_Premium'].fillna('No')
dfpRaw['Streaming_TV'] = dfpRaw['Streaming_TV'].fillna('No')
dfpRaw['Streaming_Películas'] = dfpRaw['Streaming_Películas'].fillna('No')
dfpRaw['Streaming_Música'] = dfpRaw['Streaming_Música'].fillna('No')
dfpRaw['Datos_Ilimitados'] = dfpRaw['Datos_Ilimitados'].fillna('No')

# COMMAND ----------

dfpRaw.isnull().sum()

# COMMAND ----------

#Utilitario para leer los indexadores
from pyspark.ml.feature import StringIndexerModel

# COMMAND ----------

indexadorGenero = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorGenero_EstadoClientes/")
indexadorCasado =StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorCasado_EstadoClientes/") 
indexadorServicio_Telefonico =StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorServicio_Telefonico_EstadoClientes/")
indexadorLineas_Multiples = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorLineas_Multiples_EstadoClientes/")
indexadorServicio_Internet= StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorServicio_Internet_EstadoClientes/")
indexadorTipo_Internet =StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorTipo_Internet_EstadoClientes/")
indexadorSeguridad_Linea = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorSeguridad_Linea_EstadoClientes/")
indexadorRespaldoLinea = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorRespaldoLinea_EstadoClientes/")
indexadorProteccionDispositivos = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorProteccionDispositivos_EstadoClientes/")
indexadorTipo_Soporte_Premium = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorTipo_Soporte_Premium_EstadoClientes/")
indexadorStreaming_TV = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorStreaming_TV_EstadoClientes/")
indexadorStreaming_Películas = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorStreaming_Películas_EstadoClientes/")
indexadorStreaming_Música= StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorStreaming_Música_EstadoClientes/")
indexadorDatos_Ilimitados = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorDatos_Ilimitados_EstadoClientes/")
indexadorContrato = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorContrato_EstadoClientes/")
indexadorFacturacion_sin_Papel = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorFacturacion_sin_Papel_EstadoClientes/")
indexadorMetodo_Pago = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexadorMetodo_Pago_EstadoClientes/")
indexador_Estado = StringIndexerModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/indexador_Estado/")                            

# COMMAND ----------

dfRaw1 = dfpRaw.to_spark()
dfRaw1.show()

# COMMAND ----------

dfRaw2 = indexadorGenero.transform(dfRaw1)
dfRaw3 = indexadorCasado.transform(dfRaw2)
dfRaw4 = indexadorServicio_Telefonico.transform(dfRaw3)
dfRaw5 = indexadorLineas_Multiples.transform(dfRaw4)
dfRaw6 = indexadorServicio_Internet.transform(dfRaw5)
dfRaw7 = indexadorTipo_Internet.transform(dfRaw6)
dfRaw8 = indexadorSeguridad_Linea.transform(dfRaw7)
dfRaw9 = indexadorRespaldoLinea.transform(dfRaw8)
dfRaw10 = indexadorProteccionDispositivos.transform(dfRaw9)
dfRaw11= indexadorTipo_Soporte_Premium.transform(dfRaw10)
dfRaw12 = indexadorStreaming_TV.transform(dfRaw11)
dfRaw13 = indexadorStreaming_Películas.transform(dfRaw12)
dfRaw14= indexadorStreaming_Música.transform(dfRaw13)
dfRaw15= indexadorDatos_Ilimitados.transform(dfRaw14)
dfRaw16= indexadorContrato.transform(dfRaw15)
dfRaw17= indexadorFacturacion_sin_Papel.transform(dfRaw16)
dfRaw18= indexadorMetodo_Pago.transform(dfRaw17)
display(dfRaw18)

# COMMAND ----------

# MAGIC %md
# MAGIC **Vectorizando los Features**

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

dfRaw18 = VectorAssembler(
    inputCols=[
        "Genero_Indexado",
        "Edad",
        "Casado_Indexado",
        "Número_de_Referencias",
        "Antigüedad_en_Meses",
        "Servicio_Teléfonico_Indexado",
        "Lineas_Multiples_Indexado",
        "Servicio_Internet_Indexado",
        "Tipo_Internet_Indexado",
        "SeguridadLinea_Indexado",
        "RespaldoLinea_Indexado",
        "Proteccion_Dispositivos_Indexado",
        "Soporte_Premium_Indexado",
        "Streaming_TV_Indexado",
        "Streaming_Películas_Indexado",
        "Streaming_Música_Indexado",
        "Datos_Ilimitados_Indexado",
        "Contrato_Indexado",
        "Facturación_Sin_Papel_Indexado",
        "Método_de_Pago_Indexado",
        "Cargo_Mensual",
        "Total_de_Reembolsos",
        "Total_de_Cargos_Adicionales_por_Datos",
        "Total_de_Cargos_por_Larga_Distancia"
    ],
    outputCol="features"
).transform(dfRaw18)
#Verificando
dfRaw18.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Usando el mejor modelo uu**

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassificationModel

# COMMAND ----------

modelo = RandomForestClassificationModel.load("dbfs:/FileStore/_AbandonoClientesAnalisis/output/modelo_RandomForest/")

# COMMAND ----------

df_prediccion = modelo.transform(dfRaw18)
df_prediccion.show()

# COMMAND ----------

df_prediccion.select(
    "Estado_del_Cliente",
    "prediction",
    "probability"
).show(20,False)

# COMMAND ----------

#Utilitario para des-indexar
from pyspark.ml.feature import IndexToString

# COMMAND ----------

df_prediccion = IndexToString(
  inputCol = "prediction", 
  outputCol = "Estado_del_Cliente_Prediccion", 
  labels = indexador_Estado.labels
).transform(df_prediccion)
 
#Verificamos
df_prediccion.select(
    "Estado_del_Cliente_Prediccion",
    "prediction",
    "probability"
).show(20,False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Almacenando variables originales**

# COMMAND ----------

#Seleccionamos los campos que le interesan a negocio
dfPrediccionAnalisis = df_prediccion.select(
  df_prediccion["ID_Cliente"],
  df_prediccion["Género"],
  df_prediccion["Edad"],
  df_prediccion["Casado"],
  df_prediccion["Estado"],
  df_prediccion["Número_de_Referencias"],
  df_prediccion["Antigüedad_en_Meses"],
  df_prediccion["Oferta_de_Valor"],
  df_prediccion["Servicio_Teléfonico"],
  df_prediccion["Líneas_Múltiples"],
  df_prediccion["Servicio_de_Internet"],
  df_prediccion["Tipo_de_Internet"],
  df_prediccion["Seguridad_en_Línea"],
  df_prediccion["Respaldo_en_Línea"],
  df_prediccion["Plan_de_Protección_de_Dispositivos"],
  df_prediccion["Soporte_Premium"],
  df_prediccion["Streaming_TV"],
  df_prediccion["Streaming_Películas"],
  df_prediccion["Streaming_Música"],
  df_prediccion["Datos_Ilimitados"],
  df_prediccion["Contrato"],
  df_prediccion["Facturación_Sin_Papel"],
  df_prediccion["Método_de_Pago"],
  df_prediccion["Cargo_Mensual"],
  df_prediccion["Cargos_Totales"],
  df_prediccion["Total_de_Reembolsos"],
  df_prediccion["Total_de_Cargos_Adicionales_por_Datos"],
  df_prediccion["Total_de_Cargos_por_Larga_Distancia"],
  df_prediccion["Ingresos_Totales"],
  df_prediccion["Estado_del_Cliente"],
  df_prediccion["Estado_del_Cliente_Prediccion"]
)
display(dfPrediccionAnalisis)

# COMMAND ----------

dfPrediccionAnalisis.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("dbfs:/FileStore/_AbandonoClientesAnalisis/output/dataset_AnalisisClientes/")

# COMMAND ----------

# MAGIC %md
# MAGIC **Descargando el Archivo a Analizar uwu**

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/_AbandonoClientesAnalisis/output/dataset_AnalisisClientes/"))

# COMMAND ----------

print("https://community.cloud.databricks.com/files/_AbandonoClientesAnalisis/output/dataset_AnalisisClientes/part-00000-tid-1254853723296864141-890628c1-c00d-4b75-840c-6b9f7ac6f7c7-20773-1-c000.csv")
