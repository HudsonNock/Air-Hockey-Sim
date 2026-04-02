import numpy as np

def print_actions():
    action_commands = np.load("new_data/occilation_actions_np.npy")
    idx = 0
    print(action_commands[-100:])
    for i in range(10):
        print(action_commands[-100*(i+2):-100*(i+1)])

if __name__ == "__main__":
    print_actions()
