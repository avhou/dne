from building_blocks import *
import torch

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
