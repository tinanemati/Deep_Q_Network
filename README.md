# Deep Q-Learning with Atari Pong 
A program that features training a Deep-Q Learner for Atari Pong 

  - *Note that this would normally be a larger task, implying creating an architecture, choosing hyperparameters, testing, and trying again until you get a model that works. Fortunately, this has been done for us and a set of hyperparameters and architecture that works has been selected.*
  - *Note to use a GPU while training - otherwise, training will take very long (probably days rather than hours). Use a high-RAM session if that option is available.*

### This program contains two files:
* ```Deep_Q_Network.ipynb```: This is where the code is written. 
* ```model_pretrain.pth```: A pre-trained model that isn't quite there yet. You'll need to train for about 1M more frames to get this working correctly. 

## This program fulfills the following requirements:
1. Exploitation policy:
    - Recall the exploration vs exploitation tradeoff of reinforcement learning. Below you can see that the exploration component is already taken care of. You will need to implement the exploitation part. Pass the state into the Q-learner and get the action your learner thinks is best.
    - Tip #1: You can pass a state into the learner via ```self(state)```, and you can detach the output via ```.detatch().cpu().numpy()``` so you can work with it. 
2. Compute loss:
    - Recall the loss for deep Q-Learning is given by:
        - ```(f(state, actions) - (reward + gamma * max(ftarget(next_state)))^2```
    - Tip #1: You can access the max element of each instance in the batch via:
        - ```<tensor>.detach().max(1)[0]```
    - Tip #2: Remember that you do not want to include future states if the “done” flag is true. You can figure out which states are done by multiplying the output of the target model by (1-done). 
    - Tip #3: PyTorch has an MSE loss function already implemented. Simply use:
      - ```torch.nn.MSELoss(reduction='sum')(output, target)```
    - Tip #4: You can quickly index the actions via the ```<tensor>.gather()``` command. 
3. Sample from the replay buffer:
    - In order to have diverse training instances for each update, sample batch_size frames and return a tuple of state, action, reward, next_state, done.
    - Tip #1: ```random.sample``` from the random module is useful here, as is the ```zip``` command.
4. Periodically save the model while training:
    - Don’t do this every frame, and it should save automatically after training, but it’s a good idea to save intermediate results in case your session crashes, you lose internet, etc.
    - Tip #1: There is code for saving the model at the end of the training script.




