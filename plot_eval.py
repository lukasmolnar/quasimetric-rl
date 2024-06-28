import matplotlib.pyplot as plt
import json

BASE_DIR = './online/results/gcrl_MountainCar-v0/'
RUNS = [
    'run_eps_greedy_200k',
    'run_novel_eps_greedy_200k',
    'run_pure_novel_200k',
    'run_softmax_200k',
    'run_pure_novel_state_200k_',
]

class EvalData:
    def __init__(self):
        self.env_steps = []
        self.succ_rate = []
        self.succ_rate_ts = []
        self.hitting_time = []

    def add_data(self, path):
        with open(path, 'r') as f:
            for line in f.readlines():
                line_dict = json.loads(line)
                self.env_steps.append(line_dict['env_steps'])
                self.succ_rate.append(line_dict['succ_rate'])
                self.succ_rate_ts.append(line_dict['succ_rate_ts'])
                self.hitting_time.append(line_dict['hitting_time'])

eval_dict = {}
for run in RUNS:
    eval_dict[run] = EvalData()
    path = BASE_DIR + run + '/eval.log'
    eval_dict[run].add_data(path)

# plot success rates and hitting times
fig, ax = plt.subplots(3, figsize=(10, 10))

for run, eval_data in eval_dict.items():
    ax[0].plot(eval_data.env_steps, eval_data.succ_rate, label=run)
    ax[1].plot(eval_data.env_steps, eval_data.succ_rate_ts, label=run)
    ax[2].plot(eval_data.env_steps, eval_data.hitting_time, label=run)

ax[0].set_title('Success rate')
ax[1].set_title('Success avg time')
ax[2].set_title('Hitting time')
ax[0].set_ylabel('success rate')
ax[1].set_ylabel('time')
ax[2].set_ylabel('time steps')

for i in range(3):
    ax[i].set_xlabel('env steps')
    ax[i].legend()

fig.tight_layout()

plt.savefig(BASE_DIR + 'eval.png')
