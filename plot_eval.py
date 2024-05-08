import matplotlib.pyplot as plt
import json

BASE_DIR = './online/results/gcrl_MountainCar-v0/'
RUNS = [
    'run_eps_greedy_200k',
]

class EvalData:
    def __init__(self):
        self.env_steps = []
        self.succ_rate = []
        self.hitting_time = []

    def add_data(self, path):
        with open(path, 'r') as f:
            for line in f.readlines():
                line_dict = json.loads(line)
                self.env_steps.append(line_dict['env_steps'])
                self.succ_rate.append(line_dict['succ_rate'])
                self.hitting_time.append(line_dict['hitting_time'])

eval_dict = {}
for run in RUNS:
    eval_dict[run] = EvalData()
    path = BASE_DIR + run + '/eval.log'
    eval_dict[run].add_data(path)

# plot success rates and hitting times
fig, ax = plt.subplots(2)

for run, eval_data in eval_dict.items():
    ax[0].plot(eval_data.env_steps, eval_data.succ_rate, label=run)
    ax[1].plot(eval_data.env_steps, eval_data.hitting_time, label=run)

ax[0].set_title('Success rate')
ax[1].set_title('Hitting time')
ax[0].set_xlabel('env steps')
ax[1].set_xlabel('env steps')
ax[0].set_ylabel('success rate')
ax[1].set_ylabel('hitting time')
ax[0].legend()
ax[1].legend()
fig.tight_layout()

plt.savefig(BASE_DIR + 'eval.png')
