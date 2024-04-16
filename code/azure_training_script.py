import os
import argparse
import mlflow
from azure.ai.ml.constants import AssetTypes
from building_blocks import *
import torch


def main():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="output directory")
    args = parser.parse_args()

    # Start Run
    mlflow.start_run()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frequencies = ["D", "4h"]
    layers = [2]
    heads = [4]
    forward_expansions = [256]
    for model_params, scenario_params in generate_scenarios(
        "base-transformer", device, "FourierEncoder", frequencies, layers, heads, forward_expansions
    ):
        print(f"will execute scenario {scenario_params.name}")
        model = TimeSeriesTransformer.from_params(model_params).to(device)
        scenario = Scenario(scenario_params)
        scenario.execute(model)
    print(f"execution done")

    torch.save(model, os.path.join(args.output_dir, "model.pt"))

    mlflow.end_run()


if __name__ == "__main__":
    main()
