# Databricks notebook source
from pyspark.sql.types import StructType,StructField
from pyspark.sql.types import StringType,DoubleType
import pyspark.pandas as pd
pd.set_option('display.max_rows',35)

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
).load("dbfs:/FileStore/_AbandonoClientesAnalisis/Clientes_Abandonaron_Quedaron.csv")

dfRaw.show()

# COMMAND ----------

dfpRaw = pd.from_pandas(dfRaw.toPandas())
dfpRaw

# COMMAND ----------

dfpRaw["ID_Cliente"].count()

# COMMAND ----------

dfpRaw.isnull().sum()

# COMMAND ----------

#Imputando datos con la moda, de solo las variables relevantes para el modelo
dfpRaw['Líneas_Múltiples'].fillna('No',inplace = True)
dfpRaw['Tipo_de_Internet'].fillna('Otro',inplace = True)
dfpRaw['Seguridad_en_Línea'].fillna('No',inplace = True)
dfpRaw['Respaldo_en_Línea'].fillna('No',inplace = True)
dfpRaw['Plan_de_Protección_de_Dispositivos'].fillna('No',inplace = True)
dfpRaw['Soporte_Premium'].fillna('No',inplace = True)
dfpRaw['Streaming_TV'].fillna('No',inplace = True)
dfpRaw['Streaming_Películas'].fillna('No',inplace = True)
dfpRaw['Streaming_Música'].fillna('No',inplace = True)
dfpRaw['Datos_Ilimitados'].fillna('No',inplace = True)
dfpRaw.isnull().sum()

# COMMAND ----------

dfRaw = dfpRaw.to_spark()
dfRaw.show()

# COMMAND ----------

#Indexar variables categoricas
from pyspark.ml.feature import StringIndexer 

# COMMAND ----------

indexadorGenero = StringIndexer(inputCol="Género",outputCol="Genero_Indexado").fit(dfRaw)
indexadorCasado = StringIndexer(inputCol="Casado",outputCol="Casado_Indexado").fit(dfRaw)
indexadorServicio_Telefonico = StringIndexer(inputCol="Servicio_Teléfonico",outputCol="Servicio_Teléfonico_Indexado").fit(dfRaw)
indexadorLineas_Multiples = StringIndexer(inputCol="Líneas_Múltiples",outputCol="Lineas_Multiples_Indexado").fit(dfRaw)
indexadorServicio_Internet= StringIndexer(inputCol="Servicio_de_Internet",outputCol="Servicio_Internet_Indexado").fit(dfRaw)
indexadorTipo_Internet = StringIndexer(inputCol="Tipo_de_Internet",outputCol="Tipo_Internet_Indexado").fit(dfRaw)
indexadorSeguridad_Linea = StringIndexer(inputCol="Seguridad_en_Línea",outputCol="SeguridadLinea_Indexado").fit(dfRaw)
indexadorRespaldoLinea = StringIndexer(inputCol="Respaldo_en_Línea",outputCol="RespaldoLinea_Indexado").fit(dfRaw)
indexadorProteccionDispositivos = StringIndexer(inputCol="Plan_de_Protección_de_Dispositivos",outputCol="Proteccion_Dispositivos_Indexado").fit(dfRaw)
indexadorTipo_Soporte_Premium = StringIndexer(inputCol="Soporte_Premium",outputCol="Soporte_Premium_Indexado").fit(dfRaw)
indexadorStreaming_TV = StringIndexer(inputCol="Streaming_TV",outputCol="Streaming_TV_Indexado").fit(dfRaw)
indexadorStreaming_Películas = StringIndexer(inputCol="Streaming_Películas",outputCol="Streaming_Películas_Indexado").fit(dfRaw)
indexadorStreaming_Música= StringIndexer(inputCol="Streaming_Música",outputCol="Streaming_Música_Indexado").fit(dfRaw)
indexadorDatos_Ilimitados = StringIndexer(inputCol="Datos_Ilimitados",outputCol="Datos_Ilimitados_Indexado").fit(dfRaw)
indexadorContrato = StringIndexer(inputCol="Contrato", outputCol="Contrato_Indexado").fit(dfRaw)	
indexadorFacturacion_sin_Papel = StringIndexer(inputCol="Facturación_Sin_Papel", outputCol="Facturación_Sin_Papel_Indexado").fit(dfRaw)
indexadorMetodo_Pago = StringIndexer(inputCol="Método_de_Pago",outputCol="Método_de_Pago_Indexado").fit(dfRaw)

# COMMAND ----------

dfRaw = indexadorGenero.transform(dfRaw)
dfRaw = indexadorCasado.transform(dfRaw)
dfRaw = indexadorServicio_Telefonico.transform(dfRaw)
dfRaw = indexadorLineas_Multiples.transform(dfRaw)
dfRaw = indexadorServicio_Internet.transform(dfRaw)
dfRaw = indexadorTipo_Internet.transform(dfRaw)
dfRaw = indexadorSeguridad_Linea.transform(dfRaw)
dfRaw = indexadorRespaldoLinea.transform(dfRaw)
dfRaw = indexadorProteccionDispositivos.transform(dfRaw)
dfRaw = indexadorTipo_Soporte_Premium.transform(dfRaw)
dfRaw = indexadorStreaming_TV.transform(dfRaw)
dfRaw = indexadorStreaming_Películas.transform(dfRaw)
dfRaw = indexadorStreaming_Música.transform(dfRaw)
dfRaw = indexadorDatos_Ilimitados.transform(dfRaw)
dfRaw = indexadorContrato.transform(dfRaw)
dfRaw = indexadorFacturacion_sin_Papel.transform(dfRaw)
dfRaw = indexadorMetodo_Pago.transform(dfRaw)

# COMMAND ----------

#Verificando
dfRaw.show()

# COMMAND ----------

#Indexando nuestro target
indexador_Estado = StringIndexer(inputCol="Estado_del_Cliente", outputCol="Estado_del_Cliente_indexado").fit(dfRaw)
indexador_Estado.labels

# COMMAND ----------

#Agregando el indexador al Dataframe
dfRaw = indexador_Estado.transform(dfRaw)

# COMMAND ----------

# MAGIC %md
# MAGIC **VECTORIZACION DE LOS FEACTURES**

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

dfRaw2 = VectorAssembler(
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
).transform(dfRaw)
#Verificando
dfRaw2.show()

# COMMAND ----------

#Verificamos
dfRaw2.select(
  dfRaw2["Estado_del_Cliente_indexado"],dfRaw2["features"]
).show(20, False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Guardando el Dataset Listo jeje**

# COMMAND ----------

dfRaw2.write.format("parquet").mode("overwrite").option("compression","snappy").save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/dataset_EstadoClientes/")

# COMMAND ----------

# MAGIC %md
# MAGIC **Guardando los indexadores uu**

# COMMAND ----------

indexadorGenero.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorGenero_EstadoClientes/")
indexadorCasado.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorCasado_EstadoClientes/")
indexadorServicio_Telefonico.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorServicio_Telefonico_EstadoClientes/")
indexadorLineas_Multiples.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorLineas_Multiples_EstadoClientes/")
indexadorServicio_Internet.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorServicio_Internet_EstadoClientes/")
indexadorTipo_Internet.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorTipo_Internet_EstadoClientes/")
indexadorSeguridad_Linea.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorSeguridad_Linea_EstadoClientes/")
indexadorRespaldoLinea.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorRespaldoLinea_EstadoClientes/")
indexadorProteccionDispositivos.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorProteccionDispositivos_EstadoClientes/")
indexadorTipo_Soporte_Premium.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorTipo_Soporte_Premium_EstadoClientes/")
indexadorStreaming_TV.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorStreaming_TV_EstadoClientes/")
indexadorStreaming_Películas.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorStreaming_Películas_EstadoClientes/")
indexadorStreaming_Música.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorStreaming_Música_EstadoClientes/")
indexadorDatos_Ilimitados.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorDatos_Ilimitados_EstadoClientes/")
indexadorContrato.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorContrato_EstadoClientes/")
indexadorFacturacion_sin_Papel.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorFacturacion_sin_Papel_EstadoClientes/")
indexadorMetodo_Pago.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexadorMetodo_Pago_EstadoClientes/")
indexador_Estado.write().overwrite().save("dbfs:///FileStore/_AbandonoClientesAnalisis/output/indexador_Estado/")
