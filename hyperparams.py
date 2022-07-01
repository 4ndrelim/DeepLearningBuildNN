# Hyperparams

NUM_EPOCHS = 35
LEARNING_RATE = 0.5
HIDDEN_LAYER = [100, 30] # list of layers
LAYERS = [784] + HIDDEN_LAYER + [10] # input layer -> hidden layers -> output layer (only 10 digits)
REG_PARAM = 5.0
MINI_BATCH_SIZE = 10
EARLY_STOPPING = 5 # after seeing 5 epochs without improvement to best accuracy, halt and return
