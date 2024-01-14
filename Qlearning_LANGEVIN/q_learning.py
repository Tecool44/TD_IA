import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time



def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[sprime]) - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilone):
    if random.random() < epsilone:
        var=random.randint(0, Q.shape[1] - 1)
        if var==np.argmax(Q[s]):
            return random.randint(0, Q.shape[1] - 1)
        else:
            return var
    else:
        return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1

    gamma = 0.9


    epsilon = 1

    n_epochs = 50

    max_itr_per_epoch = 50
    rewards = []

    for e in range(n_epochs):
        r = 0
        epsilon=epsilon-0.01

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            s=Sprime
            if done:
                print("done")
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    plt.plot(rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Total Reward")
    plt.title("Rewards Over Epochs")
    plt.grid(True)
    plt.show()
    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
