.PHONY: test_env_var clean blockmaze_a2c blockmaze_trainmodels blockmaze_o2a blockmaze_test_o2a
# The O2A pipeline:
# 1. Train A2C model on BlocksMaze env. Save policy weights/a2c_400000.ckpt
# 2. Use #1 policy to train:
#    a) inverse and forward model. Save models envinv_blockmaze_model.ckpt
#    b) Train inv model on BlocksMaze env. Save policy weights a2c_env_model_inverse
# 3. Use that to run O2A on a task environment with inverse model. The previously learnt forward model is used
#    as a starting point for this run, which updates its model as it goes
#    Specifically:
#    a) Use inverse model and forward model in I2A 
#    b) Relearn reward model (also forward model?)
# 4. Test the models when now get to act. Fixing the policy.
#    Note: Run tests at each point in the training... to see how much experience is needed for training to work

PYTHON=python3
TASK?=other

blockmaze_a2c:
	$(PYTHON) main.py a2c
blockmaze_trainmodels:
	#$(PYTHON) ./env_model_minigrid.py
	$(PYTHON) ./inv_model_minigrid.py
blockmaze_i2a:
	$(PYTHON) main.py i2a
blockmaze_o2a:
	$(PYTHON) main_o2a.py $(TASK)
blockmaze_test_o2a:
	$(PYTHON) test_o2a.py $(TASK)
test_env_var:
	echo "Task is $(TASK)"
clean:
	rm -rf ./weights/*

# More involved than above...
#Tests: 
#Block maze env w I2A

#Only the reward model is relearnt in obs phase -- thus what is learnt from observation is 
#the reward locations, and plans to get to the location are also made

#Why not just relearn reward function and make plans with MCTS? Can do that, but want to take
# advantage of some preceding transitions observed... need to use inverse model for that

#Todo:
#Get I2A working with blockmaze environment
#Then copy to o2a and implement O2A model. This is three things:
# - Use inverse model
# - Use 'obs' policy
# - Retrain reward model component