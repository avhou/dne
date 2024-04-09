from configparser import ConfigParser


class ConfigSettings:
    def __init__(self, config_path: str = "config.ini") -> None:
        config = ConfigParser()
        config.read(config_path)
        self.model = ModelSettings(config, "Model")
        self.data = DataSettings(config, "Data")
        self.runtime = RuntimeSettings(config, "Runtime")


class ModelSettings:
    def __init__(self, config: ConfigParser, section_name: str) -> None:
        self.batch_size = int(config.get(section_name, "BatchSize"))
        self.context_length = int(config.get(section_name, "ContextLength"))
        self.embedding_size = int(config.get(section_name, "EmbeddingSize"))
        self.num_layers = int(config.get(section_name, "NumLayers"))
        self.num_attention_heads = int(config.get(section_name, "NumAttentionHeads"))
        self.forward_expansion = int(config.get(section_name, "ForwardExpansion"))
        self.dropout = float(config.get(section_name, "Dropout"))
        self.forecast_size = int(config.get(section_name, "ForecastSize"))


class DataSettings:
    def __init__(self, config: ConfigParser, section_name: str) -> None:
        self.file_path = config.get(section_name, "SolarFilePath")
        self.retry_attempts = int(config.get(section_name, "RetryAttempts"))
        self.frequency = config.get(section_name, "Frequency")
        self.train_test_split_year = float(config.get(section_name, "TrainTestSplitYear"))
        self.train_val_split_year = float(config.get(section_name, "TrainValSplitYear"))


class RuntimeSettings:
    def __init__(self, config: ConfigParser, section_name: str) -> None:
        self.run_in_colab = config.getboolean(section_name, "RunInColab")
