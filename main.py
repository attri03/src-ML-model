from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    #print(data_ingestion.Initiate_Data_Ingestion())
    data_transformation = DataTransformation()
    print(data_transformation.Initiate_Data_Transformation(train_path=data_ingestion.data_ingestion_config.train_file_path, test_path=data_ingestion.data_ingestion_config.test_file_path))