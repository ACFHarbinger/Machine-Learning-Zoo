import unittest

from src.pipeline.inference.vllm_engine import VLLMEngine


class TestVLLMEngine(unittest.TestCase):
    def test_vllm_mock_generation(self):
        engine = VLLMEngine(model_name="facebook/opt-125m")
        prompts = ["Hello, my name is", "Capital of France is"]
        results = engine.generate(prompts)

        self.assertEqual(len(results), 2)
        for res in results:
            self.assertIsInstance(res, str)
            # Since vLLM might not be installed, it should return mock responses
            if engine.engine is None:
                self.assertTrue(res.startswith("[Mock vLLM Response"))

        print(f"\nvLLM Engine results: {results}")


if __name__ == "__main__":
    unittest.main()
