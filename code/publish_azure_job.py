from azure.ai.ml import command, Input, MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext, AmlCompute
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from utils import ConfigSettings

cf = ConfigSettings(config_path="config.ini", secrets_path="secrets.ini")

# Login to configure your workspace and resource group.
credential = DefaultAzureCredential()

# Get a handle to the workspace.
ml_client = MLClient(
    credential=credential,
    subscription_id=cf.secrets.azure.subscription_id,
    resource_group_name=cf.secrets.azure.resource_group_name,
    workspace_name=cf.secrets.azure.workspace_name,
)

env_docker_context = Environment(
    build=BuildContext(path=cf.azure.environment_path),
    name="timeseries_tranformer_env",
    description="environment to test the different time series transformer models.",
)

ml_client.environments.create_or_update(env_docker_context)

cluster = AmlCompute(
    name="ped-detection-compute",
    type="amlcompute",
    size="Standard_E4ds_v4",
    location="westeurope",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
)

ml_client.begin_create_or_update(cluster).result()

job = command(
    inputs=dict(
        output_dir="./outputs",
        training_script=cf.azure.training_script_path,
    ),
    compute=cluster.name,
    environment=env_docker_context,
    code=".",  # location of source code
    command="python ${{inputs.training_script}} --output_dir ${{inputs.output_dir}}",
    experiment_name="pytorch-timerseries-transformer-control",
    display_name="Timerseries Transformer Control",
)

# Submit the command
ml_client.jobs.create_or_update(job)
