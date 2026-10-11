
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import certifi
import numpy as np
import pandas as pd
import pymongo

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def export_collection_as_dataframe(self):
        """Read data from MongoDB."""
        try:
            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL is missing from .env")

            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=10000
            )

            self.mongo_client.admin.command("ping")

            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                raise ValueError(
                    f"No data found in {database_name}.{collection_name}"
                )

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Loaded {len(df)} records from MongoDB.")
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

        finally:
            if hasattr(self, "mongo_client"):
                self.mongo_client.close()

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True
            )

            logging.info("Data exported to feature store.")
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            training_file_path = (
                self.data_ingestion_config.training_file_path
            )
            testing_file_path = (
                self.data_ingestion_config.testing_file_path
            )

            os.makedirs(
                os.path.dirname(training_file_path),
                exist_ok=True
            )

            os.makedirs(
                os.path.dirname(testing_file_path),
                exist_ok=True
            )

            train_set.to_csv(
                training_file_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                testing_file_path,
                index=False,
                header=True
            )

            logging.info(
                f"Train records: {len(train_set)}, "
                f"test records: {len(test_set)}"
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            return DataIngestionArtifact(
                trained_file_path=(
                    self.data_ingestion_config.training_file_path
                ),
                test_file_path=(
                    self.data_ingestion_config.testing_file_path
                )
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
