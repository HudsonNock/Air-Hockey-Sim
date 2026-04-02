import numpy as np
import agent_processing as ap
import matplotlib.pyplot as plt

def print_actions():
    action_commands = np.load("new_data/occilation_actions_np.npy")
    idx = 0
    #print(action_commands[-100:])
    #for i in range(10):
    #    print(action_commands[-100*(i+2):-100*(i+1)])
        
    ap.update_path(np.array([0.23, 0.54]), np.array([0,0]), np.array([0,0]), np.array([0.5, 0.5]), np.array([1.88423576e+01, 6.18354654e+00]))
        
    recorded_pos = np.zeros((10000,3))
    action_idx = 0
    action_start = 0
    for i in range(10000):
        if i*0.0008 - action_start < action_commands[action_idx,-1]:
            pos, _, _ = ap.get_IC(i*0.0008 - action_start)
            recorded_pos[i,:] = np.concatenate([pos, np.array([i*0.0008])], axis=0)
        else:
            pos_ic, vel_ic, acc_ic = ap.get_IC(action_commands[action_idx,-1])
            data = ap.update_path(pos_ic, vel_ic, acc_ic, action_commands[action_idx,:2], action_commands[action_idx,2:4])
            
            action_start += action_commands[action_idx,-1]
            action_idx += 1
            pos, _, _ = ap.get_IC(i*0.0008 - action_start)
            recorded_pos[i,:] = np.concatenate((pos, np.array([i*0.0008])), axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recorded_pos[2000:, -1], recorded_pos[2000:, 0], color='blue', linestyle='-', linewidth=1.5)

    # Adding some polish
    plt.title('Recorded Position: First Column vs. Last Column')
    plt.xlabel('recorded_pos[:, 0]')
    plt.ylabel('recorded_pos[:, -1]')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('recorded_position_plot.png', dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":
    print_actions()
