"""run_naming: the comparison scripts must work for every instability family."""

import unittest
from pathlib import Path
import tempfile

from run_naming import (distribution_of, family_of, regime_of, pretty,
                        find_runs, families_in)


class RunNamingTests(unittest.TestCase):
    def test_every_family_is_recognised_not_just_mirror(self):
        for family in ("mirror", "firehose", "whistler"):
            name = f"{family}_bikappa3_moderate"
            self.assertEqual(family_of(name), family)
            self.assertEqual(distribution_of(name), "bikappa3")
            self.assertEqual(regime_of(name), "moderate")
            self.assertEqual(pretty(name), r"$\kappa = 3$")

    def test_unknown_directories_keep_a_non_empty_label(self):
        self.assertEqual(distribution_of("something_else"), "something_else")
        self.assertIsNone(family_of("something_else"))
        self.assertEqual(pretty("something_else"), "something_else")

    def test_isotropic_controls_are_runs_too(self):
        name = "mirror_bikappa3_isotropic"
        self.assertEqual(distribution_of(name), "bikappa3")
        self.assertEqual(regime_of(name), "isotropic")

    def test_columns_are_ordered_by_distribution_not_by_the_filesystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("firehose_bikappa5_strong", "mirror_bikappa5_moderate",
                         "mirror_bimaxwellian_moderate", "mirror_bikappa3_moderate",
                         "notes"):
                (root / name).mkdir()
            every = find_runs(root)
            self.assertEqual(families_in(every), {"mirror", "firehose"})
            mirror = [p.name for p in find_runs(root, family="mirror")]
            self.assertEqual(mirror, ["mirror_bimaxwellian_moderate",
                                      "mirror_bikappa3_moderate",
                                      "mirror_bikappa5_moderate"])
            self.assertEqual([p.name for p in find_runs(root, family="firehose")],
                             ["firehose_bikappa5_strong"])


if __name__ == "__main__":
    unittest.main()
