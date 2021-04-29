import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import time
import argparse
from adaptive_agents import NeuralModelLearner, AnalyticModelLearner, PartialModelLearner
import nonstationary_pendulum
import gym




def plot_comparison(field, fmap,  axs):
    ax0 = axs[0]
    ax0.plot(np.linspace(0, T, N), traj[:,field], c='black', alpha=0.9)
    ax0.plot(np.linspace(0, T, N), predicted_traj[:,field], c='blue', alpha=0.9)
    # for i in range(1, num_dynamics_epochs):
        # ax0.axvline(T*i/num_dynamics_epochs)
    for i in range(N):
        if drift_traj[i] == True:
            ax0.axvline(i*T/N, ymin=0.3, ymax= 0.5, c='gray',alpha=0.5)
        if adaptation_traj[i] == True:
            ax0.axvline(i*T/N, ymin=0.5, ymax= 0.7,c='cornflowerblue', alpha=0.5)
        # ax0.scatter(np.linspace(0,T,N), drift_traj)
        # ax0.scatter(np.linspace(0,T,N), adaptation_traj)

    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel(fmap[field])
    ax0.legend(["Ground Truth", "Model Prediction"])
    ax0.set_title(f"Tracking performance of {fmap[field]}")

    ax1 = axs[1]
    ax1.plot(np.linspace(0, T, N), np.abs(error_traj[:,field]), c='r')
    # for i in range(1, num_dynamics_epochs):
    #     ax1.axvline(T*i/num_dynamics_epochs)
    for i in range(N):
        if drift_traj[i] == True:
            ax1.axvline(i*T/N, ymin=0.3, ymax= 0.5, c='gray')
        if adaptation_traj[i] == True:
            ax1.axvline(i*T/N, ymin=0.5, ymax= 0.7,c='cornflowerblue')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Prediction RMSE (rad)")
    ax1.set_title(f"Error in {fmap[field]} vs. Dynamics Changes")
    ax1.legend(["Error", "Dynamics Shift"])


def loss(xnew_actual, xnew_predicted, D):
    error = np.array([((xnew_predicted[i]-xnew_actual[i])**2)**0.5 for i in range(D)])
    return error


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rate", help="learning rate for neural learner", type=float, default=0.01)
    parser.add_argument("drift_type", choices=["constant", "decaying", "oscillating", "supervised", "random"], help="type of dynamics drift")
    parser.add_argument("drift_schedule", choices=["blind", "supervised"], help="whether drift events are known")
    parser.add_argument("adaptation_strategy", choices=["detection", "supervision", "blind"])
    parser.add_argument("-num_dynamics_epochs", help="number of distinct dynamics", type=int, default=9)
    parser.add_argument("-num_control_epochs", help="number of distinct dynamics", type=int, default=500)
    parser.add_argument("-learner", help="which type of learner", choices=["neural", "analytic", "partial"], default="neural")
    parser.add_argument("--save", help="whether to display save prompt at end", action="store_true")
    parser.add_argument("-T", help="duration of experiment", type=float, default=5)
    parser.add_argument("-dt", help="time step of simulation", type=float, default=0.01)
    parser.add_argument("--render", help="whether to show trajectory during experiment", action="store_true")
    return parser




if __name__ == "__main__":
    args = setup_parser().parse_args()
    rate = args.rate
    num_dynamics_epochs = args.num_dynamics_epochs
    drift_type = args.drift_type
    drift_schedule = args.drift_schedule
    adaptation_strategy = args.adaptation_strategy
    T = args.T
    dt = args.dt
    render = args.render
    num_control_epochs = args.num_control_epochs
    N = int(T/dt)
    control_epoch_length = int(N/num_control_epochs)
    dynamics_epoch_length = int(N/num_dynamics_epochs)


    gym.envs.register(
        id='NonstationaryPendulum-v0',
        entry_point = 'nonstationary_pendulum:NonstationaryPendulumEnv',
        kwargs = {'dt':dt,'drift_speed':1.0/dynamics_epoch_length, 'drift_type':drift_type, 'schedule':drift_schedule}
    )
    env = gym.make('NonstationaryPendulum-v0')

    x0,_ = env.reset()
    u0 = env.action_space.sample()
    D = env.observation_space.shape[0]


    t0 = time.time()
    if args.learner=="analytic":
        model_learner = AnalyticModelLearner(dt=dt, memory_size=10)
    if args.learner=="neural":
        model_learner = NeuralModelLearner(rate=rate)
    if args.learner == "partial":
        model_learner = PartialModelLearner(error_thresh=1, memory_size=10, model_type=NeuralModelLearner, adaptation_strategy=adaptation_strategy, rate=rate)



    error_traj = np.empty((N,D))
    predicted_traj = np.empty((N,D))
    traj = np.empty((N,D))
    adaptation_traj = np.empty((N,))
    drift_traj = np.empty((N,))
    error_traj[0], error_traj[1] = (None, None)
    predicted_traj[0], predicted_traj[1] = (None, None)
    traj[0] = x0
    (x1, drifted), reward, done, info = env.step(u0)
    traj[1] = x1

    model_learner.observe(X=np.array([*x0, *u0]).reshape(1, -1),
                              y=np.array([*x1]).reshape(1, -1),
                          drifted = None)


    u = u0
    for i in range(2, N):
        x, drifted = env._get_obs()
        # if i % control_epoch_length == 0:
        u = env.action_space.sample()
        (xnew_actual, drifted), reward, done, info = env.step(u)
        drift_traj[i] = drifted
        traj[i] = xnew_actual
        xnew_predicted = model_learner.predict(np.array([*x,*u]).reshape(1, -1))
        predicted_traj[i] = xnew_predicted
        error = loss(xnew_actual, xnew_predicted,D)
        error_traj[i] = error
        adapting = model_learner.observe(X=np.array([*x, *u]).reshape(1, -1),
                                   y=np.array([*xnew_actual]).reshape(1, -1),
                             drifted=drifted)
        adaptation_traj[i] = adapting
        if(render):
            env.render()



    t1 = time.time()
    print(model_learner.status())
    print(f"Elapsed time (s): {t1-t0}")
    fig, axs = plt.subplots(D,2)
    fmap = env.fmap
    for i in range(D):
        plot_comparison(i, fmap, [axs[i,0], axs[i,1]])

    fig.suptitle(f"Learner: {args.learner}, Dynamics Drift: {drift_type}, Learning rate {rate}, Dynamics Epochs: {num_dynamics_epochs}")
    fig.set_size_inches(12, 12)
    plt.show()
    if args.save:
        save = input(f"Save? (Y/n):")

        if save != "n":
           leader = input("Enter leading filename (<Enter> for default):")
           fig.savefig(f"dd{drift_type}lr{rate}de{num_dynamics_epochs}.png")

    env.close()
