import os                                                                       
import multiprocessing #import Pool, Array, Process, Pipe
import time
from pynput.keyboard import Key, Controller
import pygame 
import sys
import random
import numpy as np
import _pickle as pickle
import gym

#print(os.getcwd()) #check whether terminal is in the right directory

  
                                               
def Pong(gamedata):
    pygame.init()

    #Colors
    white=(255, 255, 255)
    black=(0,0,0)

    #Screen
    screensize={"width":600, "height":400}
    screen=pygame.display.set_mode((screensize["width"], screensize["height"]))
    pygame.display.set_caption("Pong by Julius")
    screen.fill(black)

    # max games
    max_games = 11


    #display screen message
    def message_display(text):
        basicfont = pygame.font.SysFont("Bahnschrift", 24)
        text = basicfont.render(text, True, white, black)
        textrect = text.get_rect()
        textrect.centerx = screen.get_rect().centerx
        textrect.centery = screen.get_rect().centery
        screen.blit(text, textrect)

    #Objects
    class Ball(pygame.sprite.Sprite):
        

        def __init__(self, height, width, xcoordinate, ycoordinate, color, balldx, balldy, speed, player1_score, player2_score, reward):
            # Call the parent class (Sprite) constructor
            super().__init__()

            self.height=height
            self.width=width
            self.xcoordinate=xcoordinate
            self.ycoordinate=ycoordinate
            self.color=color
            self.balldx=balldx
            self.balldy=balldy
            self.speed=speed
            self.player1_score=player1_score
            self.player2_score=player2_score
            self.reward=reward
            
        def draw(self):
            pygame.draw.rect(screen, self.color, [self.xcoordinate, self.ycoordinate, self.width, self.height])
        
        def move(self):
            self.xcoordinate += self.balldx*self.speed
            self.ycoordinate += self.balldy*self.speed

        def bounce(self):
            #horizontal bounce
            if (self.ycoordinate<=0):
                self.balldy *=-1 
            if (self.ycoordinate+self.height>=screensize["height"]):
                self.balldy *=-1      

            #bounce on paddles
            #left paddle
            if (round((self.xcoordinate),0) == padleft.xcoordinate+padleft.width) and (round(self.ycoordinate,0)>=padleft.ycoordinate) and (round(self.ycoordinate,0)<=padleft.ycoordinate+padleft.height): #upper left corner bounces
                self.balldx *=-1
            elif (round((self.xcoordinate),0) == padleft.xcoordinate+padleft.width) and (round(self.ycoordinate,0)+self.height>=padleft.ycoordinate) and (round(self.ycoordinate,0)+self.height<=padleft.ycoordinate+padright.height): #lower left corner bounces
                self.balldx *=-1
            
            #rightpaddle 
            if (round((self.xcoordinate),0)+self.width == padright.xcoordinate) and (round(self.ycoordinate,0)>=padright.ycoordinate) and (round(self.ycoordinate,0)<=padright.ycoordinate+padright.height): #upper right corner bounces
                self.balldx *=-1 
            elif (round((self.xcoordinate),0)+self.width == padright.xcoordinate) and (round(self.ycoordinate,0)+self.height>=padright.ycoordinate) and (round(self.ycoordinate,0)+self.height<=padright.ycoordinate+padright.height): #lower right corner bounces
                self.balldx *=-1

        def initialize(self):
            """ needs optimisation """
            # pull ball in screen middle and give it a momentum
            self.xcoordinate = screensize["width"]/2
            self.ycoordinate = screensize["height"]/2
            self.balldx=random.uniform(-1,-0.3)*[-1,1][random.randrange(2)]
            rand=[-1,1][random.randrange(2)]
            self.balldy=np.sqrt(1-(self.balldx)*self.balldx)*rand

            #reset speed
            self.speed=speed


        def reset(self):
            # what to do when ball hits the vertical screen barriers 

            #hitting wall
            if (self.xcoordinate<=0 or self.xcoordinate>=screensize["width"]):
                #Stop ball from moving
                self.speed=0
                
                if self.xcoordinate<=0: # ball hits left bound
                    # give player 2 the point
                    self.player2_score += 1
                    #self.reward = -1 # for AI training
                    print(str(self.player1_score) + "vs" + str(self.player2_score))
                elif self.xcoordinate>=screensize["width"]: # ball hits right bound
                    # give player 1 the point
                    self.player1_score += 1
                    #self.reward = 1 # for AI training
                    print(str(self.player1_score) + "vs" + str(self.player2_score))
                
                # game ends when games played reaches max games
                if self.player1_score + self.player2_score >= max_games: 
                    if self.player1_score > self.player2_score:
                        message_display('Player 1 won. To play again press space, to quit press escape.')
                        #Update Screen
                        pygame.display.update()
                        finished = True
                        elements=[gameball.xcoordinate, gameball.ycoordinate, gameball.balldx, gameball.balldy, padleft.xcoordinate, padleft.ycoordinate, padright.xcoordinate, padright.ycoordinate, gameball.player1_score, gameball.player2_score, gameball.reward, finished]
                        for idx, ele in enumerate(elements):
                            gamedata[idx] = ele
                        while finished:
                            # game proceeeds if both Ais finish calculations
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                    # put ball back in middle and give it new momentum
                                    self.initialize()

                                    # reset scores
                                    self.player1_score = 0
                                    self.player2_score = 0
                                    #self.reward = 0
                                    finished = False
                                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                    quit()                  
                    elif self.player2_score > self.player1_score:
                        message_display('Player 2 won. To play again press space, to quit press escape.')
                        #Update Screen
                        pygame.display.update()
                        finished = True
                        elements=[gameball.xcoordinate, gameball.ycoordinate, gameball.balldx, gameball.balldy, padleft.xcoordinate, padleft.ycoordinate, padright.xcoordinate, padright.ycoordinate, gameball.player1_score, gameball.player2_score, gameball.reward, finished]
                        for idx, ele in enumerate(elements):
                            gamedata[idx] = ele
                        while finished:
                            # game proceeeds if both Ais finish calculations
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                    # put ball back in middle and give it new momentum
                                    self.initialize()

                                    # reset scores
                                    self.player1_score = 0
                                    self.player2_score = 0
                                    #self.reward = 0
                                    finished = False
                                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                                    quit() 
                    else:
                        message_display('Tie. Play one more round.')

                # continue game if not max games reached or tie
                if self.player1_score + self.player2_score < max_games or (self.player1_score + self.player2_score == max_games and self.player1_score == self.player2_score):
                    # put ball back in middle and give it new momentum
                    self.initialize()

                # Redraw all objects at current location
                screen.fill(black)
                gameball.draw()
                padleft.draw()
                padright.draw()

                #Update Screen
                pygame.display.update()

        def getreward(self):
            if self.xcoordinate<=20: # ball hits left bound
                self.reward = -1 # for AI training
            elif self.xcoordinate>=screensize["width"]-20: # ball hits right bound
                self.reward = 1 # for AI training                    
            else:
                self.reward = 0
            
            
            
    class Paddle(pygame.sprite.Sprite):

        def __init__(self,height, width, xcoordinate, ycoordinate, color, left_or_right):
            # Call the parent class (Sprite) constructor
            super().__init__()
                    
            self.height=height
            self.width=width
            self.xcoordinate=xcoordinate
            self.ycoordinate=ycoordinate
            self.color=color
            self.left_or_right=left_or_right
 

        def draw(self):
            pygame.draw.rect(screen, self.color, [self.xcoordinate, self.ycoordinate, self.width, self.height])

        def movepaddle(self):
            #move the paddles around
            key=pygame.key.get_pressed()
            
            #Movement of right paddle
            if self.left_or_right=="right":
                if key[pygame.K_LEFT]:
                    self.xcoordinate += 0
                if key[pygame.K_RIGHT]:
                    self.xcoordinate += 0
                if key[pygame.K_UP]:
                    if self.ycoordinate != 0: #only move paddle up if not hitting wall
                        self.ycoordinate += -1
                if key[pygame.K_DOWN]:
                    if self.ycoordinate+self.height < screensize["height"]: #only move paddle down bottom of rect does not hit wall
                        self.ycoordinate += 1
            
            #movement of left paddle
            if self.left_or_right=="left":
                if key[pygame.K_a]:
                    self.xcoordinate += 0
                if key[pygame.K_d]:
                    self.xcoordinate += 0
                if key[pygame.K_w]:
                    if self.ycoordinate != 0: #only move paddle up if not hitting wall
                        self.ycoordinate += -1
                if key[pygame.K_s]:
                    if self.ycoordinate+self.height < screensize["height"]: #only move paddle down bottom of rect does not hit wall
                        self.ycoordinate += 1


    #Game Values
    balldx=random.uniform(-1,-0.3)*[-1,1][random.randrange(2)]
    rand=[-1,1][random.randrange(2)]
    balldy=np.sqrt(1-(balldx)*balldx)*rand
    speed=1 #not faster than 1
    player1_score = 0
    player2_score = 0
    finished = False
    reward = 0

    gameball=Ball(20,20,screensize["width"]/2,screensize["height"]/2, white, balldx, balldy, speed, player1_score, player2_score, reward)
    padleft=Paddle(100,20,0,screensize["height"]/2-50,white, "left")
    padright=Paddle(100,20,screensize["width"]-20,screensize["height"]/2-50,white, "right")




    #GAME RUNNING
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(black)
        gameball.draw()
        padleft.draw()
        padright.draw()

        #Movement of paddles
        padleft.movepaddle()
        padright.movepaddle()

        #movement of ball
        gameball.move()
        gameball.bounce()

        #reward for AI
        gameball.getreward()

        #off screen
        gameball.reset()

        #Update Screen
        pygame.display.update()

        # Share positions with Ais
        elements=[gameball.xcoordinate, gameball.ycoordinate, gameball.balldx, gameball.balldy, padleft.xcoordinate, padleft.ycoordinate, padright.xcoordinate, padright.ycoordinate, gameball.player1_score, gameball.player2_score, gameball.reward, finished]
        for idx, ele in enumerate(elements):
            gamedata[idx] = ele

       
       






def PongAI(gamedata):
    #if __name__ == '__main__': #why do I have to include this here, but not in Pong.py?

    oldscore_player1 = 0
    oldscore_player2 = 0

    # hyperparameters
    H = 20 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # resume from previous checkpoint?

    keyboard=Controller()
    
    # model initialization
    D = len(gamedata)-4 # input dimensionality: 8 parameter of game (position and momentum of objects)
    if resume:
        model = pickle.load(open('save.p', 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)

    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

    def sigmoid(x): 
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    def policy_forward(x):
        h = np.dot(model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(model['W2'], h)
        p = sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state

    def policy_backward(eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    observation = gamedata[0:8] # not using pixels but actual position of objects, dont include score

    prev_x = None # used in computing the difference frame 
    """ May need to change this, as balldx and dy are used """
    xs,hs,dlogps,drs = [],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0

    
    while True:
    # Ai1 processes
        # set input
        cur_x = np.asarray(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D) # model uses difference in position, not actual position
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        
        # calculate next action
        if np.random.uniform() < aprob:
            keyboard.press("w")
            time.sleep(0.1)
            keyboard.release("w")
            action = 2
        else: 
            keyboard.press("s") 
            time.sleep(0.1)
            keyboard.release("s")
            
            action = 3

        # record various intermediates (needed later for backprop)
        xs.append(x) # observation
        hs.append(h) # hidden state
        y = 1 if action == 2 else 0 # a "fake label" 
        dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurement
        newscore_player1 = gamedata[8] 
        newscore_player2 = gamedata[9]                 
        reward = newscore_player1 - oldscore_player1 + oldscore_player2 - newscore_player2
        oldscore_player1 = newscore_player1
        oldscore_player2 = newscore_player2
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if gamedata[11] == 1.0: # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,hs,dlogps,drs = [],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k,v in model.items():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Ai1: resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            prev_x = None
            
            # restart game after calculations finished
            while gamedata[11]:
                    keyboard.press(Key.space) # reset game
                    time.sleep(0.5)
                    keyboard.release(Key.space)

            oldscore_player1 = 0
            oldscore_player2 = 0
            gamedata[11] = False

        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward Ai1: %f' % (episode_number, reward))



def PongAI2(gamedata):

    oldscore_player1 = 0
    oldscore_player2 = 0

    # hyperparameters
    H = 20 # number of hidden layer neurons
    batch_size = 10 # every how many episodes to do a param update?
    learning_rate = 1e-4
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
    resume = False # resume from previous checkpoint?

    keyboard=Controller()
    
    # model initialization
    D = len(gamedata)-4 # input dimensionality: 8 parameter of game (position and momentum of objects)
    if resume:
        model = pickle.load(open('save2.p', 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)

    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

    def sigmoid(x): 
        return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    def discount_rewards(r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    def policy_forward(x):
        h = np.dot(model['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(model['W2'], h)
        p = sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state

    def policy_backward(eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, model['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}

    observation = gamedata[0:8] # not using pixels but actual position of objects, dont include score
    prev_x = None # used in computing the difference frame 
    """ May need to change this, as balldx and dy are used """
    xs,hs,dlogps,drs = [],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0

    
    while True:
        # Ai processes
        cur_x = np.asarray(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D) # model uses difference in position, not actual position
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)

        # which action to take depending on the result of forward prop.
        if np.random.uniform() < aprob: 
            keyboard.press(Key.up)
            time.sleep(0.1)
            keyboard.release(Key.up)
            action = 2
        else: 
            keyboard.press(Key.down) 
            time.sleep(0.1)
            keyboard.release(Key.down)
            
            action = 3

        # record various intermediates (needed later for backprop)
        xs.append(x) # observation
        hs.append(h) # hidden state
        y = 1 if action == 2 else 0 # a "fake label" whether up or down
        dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        # step the environment and get new measurement
        newscore_player1 = gamedata[8] 
        newscore_player2 = gamedata[9]                 
        reward = newscore_player2 - oldscore_player2 + oldscore_player1 - newscore_player1 # other way around compared to Ai1 because it is Ai2
        oldscore_player1 = newscore_player1
        oldscore_player2 = newscore_player2
        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if gamedata[11] == 1.0: # an episode finished
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,hs,dlogps,drs = [],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(eph, epdlogp)
            for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k,v in model.items():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Ai2: resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(model, open('save2.p', 'wb'))
            reward_sum = 0
            prev_x = None
            
            # Only Ai1 restarts game
            #space = True
            while gamedata[11]:
                if gamedata[8] + gamedata[9] >= 11:
                    keyboard.press(Key.shift) # reset game
                    #print("Space is pressed")
                    time.sleep(0.5)
                    keyboard.release(Key.shift)
                #if gamedata[8] + gamedata[9] == 0:
                    #space = False
                

            # oldscore_player1 = 0
            # oldscore_player2 = 0
            # gamedata[11] = False
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward Ai2: %f' % (episode_number, reward))




if __name__ == '__main__':                                                                     

    #define shared memeory
    gamedata=multiprocessing.Array('f', 12) 
    
    #Start Multiprocessing
    Game=multiprocessing.Process(target=Pong, args=(gamedata,))
    AI=multiprocessing.Process(target=PongAI, args=(gamedata,))
    AI2=multiprocessing.Process(target=PongAI2, args=(gamedata,))

    Game.start()
    AI.start()
    AI2.start()

    Game.join()
    AI.join()
    AI2.join()
