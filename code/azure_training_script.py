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
    frequencies = ["D"]
    layers = [2, 4, 6]
    heads = [4, 8]
    forward_expansions = [256, 512]
    for model_params, scenario_params in generate_scenarios(
        "base-transformer",
        device,
        "AsymPaddingConv1dEncoder",
        frequencies,
        layers,
        heads,
        forward_expansions,
        args.output_dir,
    ):
        print(f"will execute scenario {scenario_params.name}")
        model = TimeSeriesTransformer.from_params(model_params).to(device)
        scenario = Scenario(scenario_params)
        scenario.execute(model)

    torch.save(model, os.path.join(args.output_dir, "model.pt"))

    mlflow.end_run()


if __name__ == "__main__":
    main()
