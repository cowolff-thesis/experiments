import unittest
from unittest.mock import patch
from ThesisPackage.Environments.multi_pong_sender_receiver import PongEnvSenderReceiver

class TestPongEnv(unittest.TestCase):

    def test_pong_env(self):
        env = PongEnvSenderReceiver()
        self.assertEqual(env.width, 20)
        self.assertEqual(env.height, 10)
        self.assertEqual(env.paddle_height, 3)
        self.assertEqual(env.sequence_length, 1)
        self.assertEqual(env.vocab_size, 2)
        self.assertEqual(env.max_episode_steps, 1024)
        self.assertTrue(env.self_play)
        self.assertIsNone(env.receiver)
        self.assertEqual(env.mute_method, "random")

    def test_reset(self):
        env = PongEnvSenderReceiver()
        obs, infos = env.reset()
        self.assertEqual(obs["paddle_1"].shape, env.observation_space("paddle_1").shape)
        self.assertEqual(obs["paddle_2"].shape, env.observation_space("paddle_2").shape)

    def test_step(self):
        env = PongEnvSenderReceiver()
        env.reset()
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)

    def test_spaces(self):
        env = PongEnvSenderReceiver(self_play=False, receiver="paddle_1")
        self.assertNotEqual(env.observation_space("paddle_1").shape, env.observation_space("paddle_2").shape)
        self.assertNotEqual(env.action_space("paddle_1").shape, env.action_space("paddle_2").shape)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        self.assertEqual(obs["paddle_1"].shape, env.observation_space("paddle_1").shape)
        self.assertEqual(obs["paddle_2"].shape, env.observation_space("paddle_2").shape)

        env = PongEnvSenderReceiver(self_play=True, receiver="paddle_2")
        self.assertEqual(env.receiver, "paddle_2")
        self.assertEqual(env.mute_method, "random")
        self.assertEqual(env.observation_space("paddle_2").shape, env.observation_space("paddle_1").shape)
        self.assertEqual(env.action_space("paddle_2").shape, env.action_space("paddle_1").shape)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        self.assertEqual(obs["paddle_1"].shape, env.observation_space("paddle_1").shape)
        self.assertEqual(obs["paddle_2"].shape, env.observation_space("paddle_2").shape)

    def test_language_channel(self):
        env = PongEnvSenderReceiver(self_play=True, receiver="paddle_2")
        self.assertEqual(env.receiver, "paddle_2")
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        self.assertEqual(actions["paddle_1"][1:], obs["paddle_2"][10:])
        
        env = PongEnvSenderReceiver(self_play=False, receiver="paddle_1", mute_method="zero")
        self.assertEqual(env.receiver, "paddle_1")
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
        self.assertEqual(actions["paddle_2"][1:], obs["paddle_1"][10:])
        self.assertEqual([0 for _ in range(env.sequence_length)], obs["paddle_1"][10:])

if __name__ == '__main__':
    unittest.main()