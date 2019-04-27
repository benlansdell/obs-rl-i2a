.PHONY: test_env_var clean check-env blockmaze_a2c blockmaze_trainmodels blockmaze_o2a blockmaze_test_o2a
# The O2A pipeline:
# 1. Train A2C model on BlocksMaze env. Save policy weights/a2c_400000.ckpt
# 2. Use #1 policy to train inverse and forward model. Save models envinv_blockmaze_model.ckpt
# 3. Use that to run O2A on a task environment with inverse model. The previously learnt forward model is used as a starting point for this run, which updates its model as it goes
# 4. Test the models when now get to act. Fixing the policy.
#    Note: Run tests at each point in the training... to see how much experience is needed for training to work

PYTHON=python3
TASK?=other

blockmaze_a2c:
	$(PYTHON) main.py a2c
blockmaze_trainmodels:
	$(PYTHON) ./env_model_minigrid.py
blockmaze_o2a:
	$(PYTHON) main_o2a.py $(TASK)
blockmaze_test_o2a:
	$(PYTHON) test_o2a.py $(TASK)
test_env_var:
	echo "Task is $(TASK)"
clean:
	rm -rf ./weights/*