from configparser import ConfigParser


class ConfigSettings:
    def __init__(self, config_path: str = "config.ini", secrets_path: str = None) -> None:
        config = ConfigParser()
        config.read(config_path)
        self.model = ModelSettings(config, "Model")
        self.data = DataSettings(config, "Data")
        self.runtime = RuntimeSettings(config, "Runtime")
        self.azure = AzureSettings(config, "Azure")
        if secrets_path:
            self.secrets = Secrets(secrets_path)


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
        self.encoder_type = str(config.get(section_name, "EncoderType"))
        self.kernel_size = int(config.get(section_name, "KernelSize"))
        self.padding_right = int(config.get(section_name, "PaddingRight"))


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
        self.run_on_gpu = config.getboolean(section_name, "RunOnGPU")


class AzureSettings:
    def __init__(self, config: ConfigParser, section_name: str) -> None:
        self.environment_path = config.get(section_name, "EnvironmentPath")
        self.training_script_path = config.get(section_name, "TrainingScriptPath")


class AzureSecrets:
    def __init__(self, config: ConfigParser, section_name: str) -> None:
        self.subscription_id = config.get(section_name, "SubscriptionId")
        self.resource_group_name = config.get(section_name, "ResourceGroupName")
        self.workspace_name = config.get(section_name, "WorkspaceName")


class Secrets:
    def __init__(self, secrets_path: str = "secrets.ini") -> None:
        secrets = ConfigParser()
        secrets.read(secrets_path)
        self.azure = AzureSecrets(secrets, "Azure")
