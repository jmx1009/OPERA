"""Load the two OPERA libraries for fresh training, resume and evaluation."""

from pathlib import Path

from .insight_library import InsightLibrary


def load_previous_round_library(config, artifacts, round_number, tasks):
    """Load the two stored partitions which supply a numbered refinement round."""
    if round_number < 1:
        raise ValueError("Numbered refinement rounds start at one")
    suffix = "base" if round_number == 1 else f"refine_iter{round_number - 1}"
    directory = Path(config.file_paths.lib_dir)
    return load_working_library(
        config, artifacts, directory / f"library_{suffix}.json",
        directory / f"latest_taxonomy_{suffix}.json", historical_tasks=tasks,
    )


def load_working_library(config, artifacts, library_path=None, taxonomy_path=None, *,
                         historical_tasks=(), independent_experience=False):
    library = (InsightLibrary.from_json_file(library_path, taxonomy_path)
               if library_path else InsightLibrary())
    if independent_experience and library_path:
        joint_lessons = Path(__file__).resolve().parents[1] / "data/libraries/lessons.json"
        if library.principles or Path(library_path).samefile(joint_lessons):
            raise ValueError("Experience library only requires lessons learned independently of principles.")
    library.reserve_historical_ids(historical_tasks)
    bootstrap = getattr(config, "bootstrap", None)
    if not artifacts:
        if bootstrap is not None and not bootstrap.enable_principle_library:
            library.select_types({"lesson"})
        return library
    # A versioned checkpoint owns both partitions, including an intentionally
    # empty one. Its conditions and evidence take precedence over bootstrap files.
    if library.is_checkpoint:
        return library
    principles = InsightLibrary.from_json_file(
        bootstrap.consolidated_principles_path, bootstrap.principle_taxonomy_path
    )
    library.attach_principles(principles)
    return library
