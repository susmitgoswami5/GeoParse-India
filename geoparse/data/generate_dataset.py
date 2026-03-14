#!/usr/bin/env python3
"""
CLI script to generate synthetic Indian address NER training data.

Usage:
    python -m geoparse.data.generate_dataset [--num-samples 50000] [--output-dir data/]
"""

import argparse
import sys

from geoparse.data.synthetic_generator import SyntheticAddressGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indian address NER training data"
    )
    parser.add_argument(
        "--num-samples", type=int, default=50000,
        help="Total number of samples to generate (default: 50000)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/",
        help="Output directory for dataset files (default: data/)"
    )
    parser.add_argument(
        "--noise-prob", type=float, default=0.6,
        help="Probability of applying noise to each sample (default: 0.6)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.85,
        help="Train/validation split ratio (default: 0.85)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GeoParse-India: Synthetic Data Generator")
    print("=" * 60)
    print(f"  Samples:     {args.num_samples}")
    print(f"  Noise Prob:  {args.noise_prob}")
    print(f"  Train Ratio: {args.train_ratio}")
    print(f"  Output Dir:  {args.output_dir}")
    print(f"  Seed:        {args.seed}")
    print("=" * 60)

    generator = SyntheticAddressGenerator(
        noise_probability=args.noise_prob,
        seed=args.seed,
    )

    train_path, val_path = generator.save_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
    )

    # Print sample addresses
    print("\n--- Sample Generated Addresses ---")
    gen2 = SyntheticAddressGenerator(noise_probability=args.noise_prob, seed=args.seed + 1)
    for i in range(5):
        sample = gen2.generate_sample()
        print(f"\n[{i+1}] {'(Noisy)' if sample['is_noisy'] else '(Clean)'}")
        print(f"    Text:   {sample['text']}")
        print(f"    Tokens: {sample['tokens']}")
        print(f"    Labels: {sample['labels']}")

    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()
