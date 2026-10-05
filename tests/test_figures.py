from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib.figure import Figure

from mtpv_optimizer.figures import apply_publication_style, save_figure_bundle


class FigureExportTests(unittest.TestCase):
    def test_publication_bundle_has_editable_svg_text(self) -> None:
        apply_publication_style("Arial")
        figure = Figure(figsize=(3.0, 2.0))
        axes = figure.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_xlabel("Pressure drop")
        axes.set_ylabel("Output power")
        with TemporaryDirectory() as directory:
            paths = save_figure_bundle(figure, Path(directory) / "qa")
            self.assertEqual({path.suffix for path in paths}, {".svg", ".pdf", ".tiff", ".png"})
            svg = (Path(directory) / "qa.svg").read_text(encoding="utf-8")
            self.assertIn("<text", svg)
            self.assertIn("Pressure drop", svg)


if __name__ == "__main__":
    unittest.main()
