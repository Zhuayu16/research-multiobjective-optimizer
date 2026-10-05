"""Configurable engineering optimization with a retained MTPV/MTEG workflow."""

__version__ = "0.2.0"

from .core import ModelBundle, fit_surrogates, load_table, validate_dataset
from .dat_parser import DatImportResult, import_dat_files, parse_dat_file
from .doe import central_composite_design
from .nsga2 import NSGA2Result, optimize_nsga2

__all__ = [
    "ModelBundle",
    "NSGA2Result",
    "DatImportResult",
    "central_composite_design",
    "fit_surrogates",
    "import_dat_files",
    "load_table",
    "optimize_nsga2",
    "parse_dat_file",
    "validate_dataset",
]
