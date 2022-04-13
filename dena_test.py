

from gym_folder.alphartc_gym import gym_file

total_stats = []
g = gym_file.Gym("test_dena")
g.reset()
while True:
    stats, done = g.step(1000)
    print(stats)
    if not done:
        total_stats += stats
    else:
        break
assert (total_stats)
for stats in total_stats:
    assert (isinstance(stats, dict))