# Pong-Ai
This is one of my first python projects. It sets up a self programmed instance of Pong and trains two Q-Learning AIs to play the game against each other.

Using the coordinates of the two paddles, the position and momentum of the ball and information about the current score, two AIs using Q-Learning are trained to play Pong against each other. Raw pixel input is not used as this increases training time dramatically. The Q-Learning algorithm is based on Andrej Karpathy's blog post on "Deep Reinforcement Learning" which can be accessed here: http://karpathy.github.io/2016/05/31/rl/.

The Pong instance is programmed with pygame and is controlled via keyboard inputs. Meaning the game can be played by humans and AIs. 

In order to run Pong and both AIs simultaneously, the multiprocessing library of python is used. 
