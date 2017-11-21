from ple.games.flappybird import FlappyBird
from ple import PLE

from RandomAgent import RandomAgent

# The doc of Flappy Bird is at
# http://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html.
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
agent = RandomAgent(actions=p.getActionSet())

p.init()
reward = 0.0
nb_frames = 200

for i in range(nb_frames):
    if p.game_over():
        p.reset_game()

    observation = p.getScreenRGB()
    action = agent.pickAction(reward, observation)
    reward = p.act(action)
