

from apply_model import BandwidthEstimator
from apply_model import BandwidthEstimator_hrcc
from apply_model import BandwidthEstimator_gcc

model_path_dena = "./model/ppo_2022_05_05_16_19_31_v2.pth"

states = {
    "send_time_ms": 100,
    "arrival_time_ms": 400,
    "payload_type": 125,
    "sequence_number": 10,
    "ssrc": 123,
    "padding_length": 0,
    "header_length": 120,
    "payload_size": 1350
}

# Challenge example estimator - an RF model using a trained model "pretrained_model.pth"
# model_path = "./model/pretrained_model.pth"
model_path = "./model/ppo_2022_05_05_16_19_31_v2.pth"
BWE = BandwidthEstimator.Estimator(model_path)
bandwidth_prediction1 = int(BWE.get_estimated_bandwidth())
print("Reality check step time: ", BWE.step_time)
print("Prediction from challenge example estimator", bandwidth_prediction1)
print("----------------------------------------------")

# hrcc_model_path = './model/ppo_2021_07_25_04_57_11_with500trace.pth'
hrcc_model_path = "./model/ppo_2022_05_05_16_19_31_v2.pth"
BWE_hrcc = BandwidthEstimator_hrcc.Estimator(hrcc_model_path)
bandwidth_prediction2 = BWE_hrcc.get_estimated_bandwidth()
print("Reality check step time: ", BWE_hrcc.step_time)
print("Prediction from HRCC estimator", bandwidth_prediction2)
print("----------------------------------------------")

BWE_gcc = BandwidthEstimator_gcc.GCCEstimator()
bandwidth_prediction3, _ = BWE_gcc.get_estimated_bandwidth()
print("Reality check state: ", BWE_gcc.state)
print("Prediction fromGCC estimator", bandwidth_prediction3)

