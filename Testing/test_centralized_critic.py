import unittest
from unittest.mock import patch
from ThesisPackage.Environments.multi_pong_language import PongEnv

class TestCentralizedCritic(unittest.TestCase):
    def test_pong_env(self):
        env = PongEnv()
        self.assertEqual(env.width, 20)
        self.assertEqual(env.height, 10)
        self.assertEqual(env.paddle_height, 3)
        self.assertEqual(env.sequence_length, 1)
        self.assertEqual(env.vocab_size, 2)
        self.assertEqual(env.max_episode_steps, 1024)
        
        _, _ = env.reset()
        state = env.state()

        self.assertEqual(env.observation_space("critic").shape, state.shape)

if __name__ == "__main__":
    unittest.main()