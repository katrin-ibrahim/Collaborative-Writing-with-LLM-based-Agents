#!/usr/bin/env python3
import argparse
import yaml
from agents.writer import WriterAgent
from utils.data_loader import load_wildseek


def main():
    parser = argparse.ArgumentParser(description="Run the WriterAgent on WildSeek data.")
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--writer_config", default="configs/writer.yaml")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    # Load configs
    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.writer_config) as f:
        writer_cfg = yaml.safe_load(f)
    # Merge
    config = {**base_cfg, **writer_cfg}

    # Load data
    ds_cfg = writer_cfg['dataset']
    samples = load_wildseek(
        ds_cfg['name'], ds_cfg.get('config'), ds_cfg['split'], ds_cfg['text_field']
    )[: args.num_samples]

    # Initialize agent
    agent = WriterAgent(config)

    # Generate and print
    for idx, input_text in enumerate(samples, 1):
        output = agent.generate(input_text)
        print(f"=== Sample {idx} ===")
        print("Input:", input_text)
        print("Output:", output)
        print()

if __name__ == "__main__":
    main()