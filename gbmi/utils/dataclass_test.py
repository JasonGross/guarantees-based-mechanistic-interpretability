from dataclasses import dataclass, fields
from typing import Literal, Union

from gbmi.utils.dataclass import enumerate_dataclass_values, get_values_of_type
from gbmi.utils.testing import TestCase


class ConfigTest(TestCase):
    @dataclass
    class ExampleConfig:
        a: Union[None, bool, Literal[1, 2]]
        b: Union[bool, Literal["x", "y"]]

    def test_get_values_of_type(self):
        self.assertEqual(
            set(get_values_of_type(fields(self.ExampleConfig)[0].type)),
            set([None, True, False, 1, 2]),
        )

        self.assertEqual(
            set(get_values_of_type(fields(self.ExampleConfig)[1].type)),
            set([True, False, "x", "y"]),
        )

    def test_enumerate_dataclass_values(self):
        # This is where you would replace ExampleConfig with your actual dataclass,
        # like HookedTransformerConfig, MaxOfN, etc.
        all_configs = list(enumerate_dataclass_values(self.ExampleConfig))

        # Here you can write assertions about the generated configurations
        # For simplicity, we'll just check the count of generated configurations
        expected_count = 20  # Based on the possible values in ExampleConfig
        actual_count = len(all_configs)

        # If you want to check the integrity of each configuration, you can loop through them
        # and make assertions as needed, for example:
        for config in all_configs:
            self.assertIn(
                config.a, [None, True, False, 1, 2], "Invalid value for field 'a'"
            )
            self.assertIn(
                config.b, [True, False, "x", "y"], "Invalid value for field 'b'"
            )

        for a in [None, True, False, 1, 2]:
            for b in [True, False, "x", "y"]:
                self.assertIn(
                    self.ExampleConfig(a, b),
                    all_configs,
                    f"Missing configuration: ({a}, {b})",
                )

        self.assertEqual(
            actual_count,
            expected_count,
            f"The number of generated configurations does not match the expected count: {[(c.a, c.b) for c in all_configs[5:]]}",
        )
