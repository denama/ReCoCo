

# from gym_folder.alphartc_gym import gym_file
#
# total_stats = []
# g = gym_file.Gym("test_dena")
# g.reset()
#
# while True:
#     stats, done = g.step(10)
#     if stats:
#         print("Stats", stats)
#     if not done:
#         total_stats += stats
#     else:
#         break

# assert (total_stats)
# for stats in total_stats:
#     assert (isinstance(stats, dict))

import subprocess
import os
import signal

__ROOT_PATH__ = os.path.dirname(os.path.abspath(__file__))
__GYM_PROCESS_PATH__ = os.path.join(__ROOT_PATH__, "gym_folder", "target", "gym")


class GymProcessDena(object):
    def __init__(
        self,
        gym_id: str = "gym",
        trace_path: str = "",
        report_interval_ms: int = 60,
        duration_time_ms: int = 3000):
        process_args = [__GYM_PROCESS_PATH__,]
        process_args.append("--gym_id="+str(gym_id))
        if trace_path:
            process_args.append("--trace_path="+trace_path)
        if report_interval_ms:
            process_args.append("--report_interval_ms="+str(report_interval_ms))
        if duration_time_ms:
            process_args.append("--duration_time_ms="+str(duration_time_ms))

        # process_args = [__GYM_PROCESS_PATH__, f"--gym_id={gym_id} --trace_path={trace_path} --report_interval_ms={report_interval_ms} --duration_time_ms={duration_time_ms}"]

        print("TRACE PATH", trace_path)
        print("COMMAND", " ".join(process_args))
        output_folder = os.path.join(__ROOT_PATH__, "simulation_analysis", "outputs")
        output_file_name = os.path.basename(trace_path).split(".")[0]
        out_full_path = os.path.join(output_folder, f'{output_file_name}_duration_{duration_time_ms}_output.txt')
        outfile = open(out_full_path, 'w')
        # o, e = subprocess.Popen(" ".join(process_args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, errors="ignore").communicate()
        # print(o)

        self.gym = subprocess.Popen(" ".join(process_args), stdout=outfile, shell=True)




    # def wait(self, timeout = None):
    #     return self.gym.wait(timeout)
    #
    # def __del__(self):
    #     self.gym.send_signal(signal.SIGINT)
    #     self.gym.send_signal(signal.SIGKILL)



if __name__ == "__main__":

    trace_file = "/home/dena/Documents/Gym_RTC/gym-example/gym_folder/alphartc_gym/tests/data/trace_example.json"
    # trace_file = "/home/dena/Documents/Gym_RTC/gym-example/traces/trace_200k.json"
    # trace_file = "/home/dena/Documents/Gym_RTC/gym-example/gym_folder/alphartc_gym/tests/data/5G_12mbps.json"

    prc = GymProcessDena(trace_path=trace_file)