"""All tunable settings in one place."""

# ---- board -------------------------------------------------------------
GRID_W = 15
GRID_H = 15

# ---- Q-learning hyperparameters ---------------------------------------
ALPHA = 0.1          # learning rate: how far each update moves toward the target
GAMMA = 0.9          # discount factor: how much future reward matters
EPS_START = 1.0      # epsilon: probability of a random (exploring) move
EPS_MIN = 0.01
EPS_DECAY = 0.995    # epsilon is multiplied by this after every episode
EPISODES = 3000      # default length of headless training

# ---- rewards -----------------------------------------------------------
R_FOOD = 10.0
R_DEATH = -10.0
R_STEP = 0.0

# Episode is cut (counts as a death) if the snake goes this many steps
# per body segment without eating. Stops it circling forever.
TIMEOUT_FACTOR = 100

# ---- state / action space ---------------------------------------------
N_BITS = 7           # 7 yes/no facts: 3 danger + 4 food direction (relative to heading)
N_STATES = 2 ** N_BITS   # -> 128 possible situations
N_ACTIONS = 3        # 0 = straight, 1 = turn right, 2 = turn left

# ---- misc -------------------------------------------------------------
SEED = 42
Q_TABLE_PATH = "q_table.npz"
RESULTS_DIR = "results"
