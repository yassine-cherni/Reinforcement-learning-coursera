# test_mdp_framework.py
import unittest
from mdp_framework import MDPFramework
from mdp_components import State, Action

class TestMDPFramework(unittest.TestCase):
    def setUp(self):
        self.mdp_framework = MDPFramework()
        # Add any additional setup for your MDP framework

    def test_transition(self):
        # Define test states and action
        state_a = State("A")
        state_b = State("B")
        action_x = Action("X")

        # Test transition
        next_state = self.mdp_framework.transition(state_a, action_x)
        self.assertEqual(next_state, state_b, "Transition did not produce the expected result")

if __name__ == '__main__':
    unittest.main()
  
