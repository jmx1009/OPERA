"""Build a document hierarchy and its candidate principles from a local corpus."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from src.principle_library import PrincipleLibraryBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Directory containing PDF, text or Markdown sources.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/phase_one"))
    parser.add_argument("--config", type=Path, default=ROOT / "train_config.yaml")
    args = parser.parse_args()
    if not args.corpus.is_dir():
        parser.error(f"Corpus directory does not exist: {args.corpus}")
    config = OmegaConf.load(args.config)
    config.bootstrap.corpus_root = str(args.corpus.resolve())
    config.bootstrap.document_index_path = str(args.output / "summaries.json")
    config.bootstrap.candidate_principles_path = str(args.output / "candidates.json")
    config.bootstrap.candidate_taxonomy_path = str(args.output / "candidate_taxonomy.json")
    builder = PrincipleLibraryBuilder(config, config.model, config.service)
    builder.build()


if __name__ == "__main__":
    main()
