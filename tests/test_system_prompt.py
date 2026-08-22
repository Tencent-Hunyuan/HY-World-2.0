import importlib.util
from pathlib import Path
import unittest


def load_system_prompt_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "hyworld2"
        / "panogen"
        / "hunyuan_image_3"
        / "system_prompt.py"
    )
    spec = importlib.util.spec_from_file_location("system_prompt", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DynamicSystemPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system_prompt = load_system_prompt_module()

    def test_combined_recaption_uses_thinking_prompt(self):
        prompt = self.system_prompt.get_system_prompt("dynamic", "think_recaption")

        self.assertEqual(
            prompt,
            self.system_prompt.t2i_system_prompts["en_think_recaption"][0],
        )

    def test_existing_dynamic_modes_still_resolve(self):
        cases = [
            ("think", self.system_prompt.t2i_system_prompts["en_think_recaption"][0]),
            ("recaption", self.system_prompt.t2i_system_prompts["en_recaption"][0]),
            ("image", self.system_prompt.t2i_system_prompts["en_vanilla"][0].strip("\n")),
        ]

        for task, expected in cases:
            with self.subTest(task=task):
                self.assertEqual(
                    self.system_prompt.get_system_prompt("dynamic", task),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
