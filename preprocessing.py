#@title EpitopeProcessor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, udf, arrays_zip, explode, mean, stddev
from pyspark.sql.types import StringType, IntegerType, FloatType, ArrayType
from pyspark.sql.functions import array
import pandas as pd
import os

class EpitopeProcessor:
    def __init__(self, positif_csv, negatif_csv, output_file):
        self.spark = SparkSession.builder.appName("EpitopePreprocessing").getOrCreate()
        self.positif_csv = positif_csv
        self.negatif_csv = negatif_csv
        self.output_file = output_file
        self._register_udfs()

    def _register_udfs(self):
        # UDF untuk preprocessing protein
        @udf(StringType())
        def protein_preprocessing(seq):
            aa = set("ACDEFGHIKLMNPQRSTVWY")
            return "".join([c for c in seq if c.upper() in aa]) if seq else ""
        self.protein_preprocessing = protein_preprocessing

        @udf(ArrayType(StringType()))
        def split_amino(seq):
            return list(seq) if seq else []
        self.split_amino = split_amino

        @udf(ArrayType(StringType()))
        def build_label(seq, label):
            return [label] * len(seq) if seq else []
        self.build_label = build_label

        @udf(ArrayType(IntegerType()))
        def build_position(seq, start):
            if seq is None or start is None:
                return []
            try:
                return [int(start) + i for i in range(len(seq))]
            except Exception:
                return []
        self.build_position = build_position

        @udf(ArrayType(IntegerType()))
        def build_length_sequence_array(seq, length):
            if seq is None or length is None:
                return []
            return [length] * len(seq)
        self.build_length_sequence_array = build_length_sequence_array

        simple_dict = {'A' : 1, 'C' : 2, 'D' : 3, 'E' : 4, 'F' : 5, 'G' : 6, 'H' : 7, 'I' : 8, 'K' : 9, 'L' : 10, 'M' : 11, 'N' : 12, 'P' : 13, 'Q' : 14, 'R' : 15, 'S' : 16, 'T' : 17, 'V' : 18, 'W' : 19, 'Y' : 20}

        @udf(IntegerType())
        def get_simple_dict_tokenizer(aa): return simple_dict.get(aa, 0.0)

        self.get_simple_dict_tokenizer = get_simple_dict_tokenizer

    def run(self):
        pos = self.spark.read.csv(self.positif_csv, header=True, inferSchema=True).withColumn("label", lit("E"))
        neg = self.spark.read.csv(self.negatif_csv, header=True, inferSchema=True).withColumn("label", lit("."))

        @udf(StringType())
        def create_actual(seq, label): return label * len(seq) if seq else ""
        pos = pos.withColumn("actual", create_actual(col("Epitope - Name"), col("label")))
        neg = neg.withColumn("actual", create_actual(col("Epitope - Name"), col("label")))

        combined = pos.select("Epitope - Starting Position", "Epitope - Ending Position", "Epitope - Name", "label", "actual") \
            .union(neg.select("Epitope - Starting Position", "Epitope - Ending Position", "Epitope - Name", "label", "actual")) \
            .withColumn("Epitope - Name", self.protein_preprocessing(col("Epitope - Name"))) \
            .withColumn("Epitope - Starting Position", col("Epitope - Starting Position").cast("int")) \
            .withColumn("Epitope - Ending Position", col("Epitope - Ending Position").cast("int")) \
            .withColumn("length_sequence", (col("Epitope - Ending Position") - col("Epitope - Starting Position")).cast("int"))


        expanded = combined \
            .withColumn("amino", self.split_amino(col("Epitope - Name"))) \
            .withColumn("label_exp", self.build_label(col("Epitope - Name"), col("label"))) \
            .withColumn("position", self.build_position(col("Epitope - Name"), col("Epitope - Starting Position")))

        exploded = exploded = expanded \
    .withColumn("length_sequence_arr", self.build_length_sequence_array(col("Epitope - Name"), col("length_sequence"))) \
    .select("amino", "label_exp", "position", "length_sequence_arr") \
    .withColumn("zipped", arrays_zip("amino", "label_exp", "position", "length_sequence_arr")) \
            .withColumn("exploded", explode(col("zipped"))) \
            .select(
                col("exploded.amino").alias("amino"),
                col("exploded.label_exp").alias("label"),
                col("exploded.position").alias("Position").cast("int"),
                col("exploded.length_sequence_arr").alias("length_sequence")
            )

        enriched = exploded \
            .withColumn("numerical_amino_acid", self.get_simple_dict_tokenizer(col("amino")))

        # Simpan hasil
        temp_dir = "temp_export"
        enriched.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)

        csv_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".csv")]
        df_pandas = pd.read_csv(csv_files[0])
        df_pandas.to_csv(self.output_file, index=False)

        print(f"✅ Output berhasil disimpan ke: {self.output_file}")


processor = EpitopeProcessor(
    positif_csv="positif.csv",
    negatif_csv="negatif.csv",
    output_file="clean_amino_data.csv"
)
processor.run()
