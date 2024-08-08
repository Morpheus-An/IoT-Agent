
from imports import *
import pandas as pd
import numpy as np
import pywt
from torch.utils.data import Dataset, DataLoader



#---------------------------------data calibrate for wifi occupancy---------------------------------
def calibrate_single_phase(phases):
    """
    Calibrate phase data from the single time moment
    Based on:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sys031fp.pdf
        https://github.com/ermongroup/Wifi_Activity_Recognition/.../phase_calibration.m

    :param phases: phase in the single time moment, np.array of shape(1, num of subcarriers)
    :return: calibrate phase, np.array of shape(1, num of subcarriers)
    """

    phases = np.array(phases)
    difference = 0

    calibrated_phase, calibrated_phase_final = np.zeros_like(phases), np.zeros_like(phases)
    calibrated_phase[0] = phases[0]

    phases_len = phases.shape[0]

    for i in range(1, phases_len):
        temp = phases[i] - phases[i - 1]

        if abs(temp) > np.pi:
            difference = difference + 1 * np.sign(temp)

        calibrated_phase[i] = phases[i] - difference * 2 * np.pi

    k = (calibrated_phase[-1] - calibrated_phase[0]) / (phases_len - 1)
    b = np.mean(calibrated_phase)

    for i in range(phases_len):
        calibrated_phase_final[i] = calibrated_phase[i] - k * i - b

    return calibrated_phase_final


def calibrate_phase(phases):
    """
    Calibrate phase data based on the following method:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/sys031fp.pdf
        https://github.com/ermongroup/Wifi_Activity_Recognition/.../phase_calibration.m

    :param phases: np.array of shape(data len, num of subcarries)
    :return: calibrated phases: np.array of shape(data len, num of subcarriers)
    """

    calibrated_phases = np.zeros_like(phases)

    for i in range(phases.shape[0]):
        calibrated_phases[i] = calibrate_single_phase(np.unwrap(phases[i]))

    return calibrated_phases


def calibrate_amplitude(amplitudes, rssi=1):
    """
    Simple amplitude normalization, that could be multiplied by rsii
    ((data - min(data)) / (max(data) - min(data))) * rssi

    :param amplitudes: np.array of shape(data len, num of subcarriers)
    :param rssi: number
    :return: normalized_amplitude: np.array of shape(data len, num of subcarriers)
    """

    amplitudes = np.array(amplitudes)
    return ((amplitudes - np.min(amplitudes)) / (np.max(amplitudes) - np.min(amplitudes))) * rssi


def calibrate_amplitude_custom(amplitudes, min_val, max_val, rssi=1):
    amplitudes = np.array(amplitudes)
    return ((amplitudes - min_val) / (max_val - min_val)) * rssi


def dwn_noise(vals):
    data = vals.copy()
    threshold = 0.06  # Threshold for filtering

    w = pywt.Wavelet('sym5')
    maxlev = pywt.dwt_max_level(data.shape[0], w.dec_len)

    coeffs = pywt.wavedec(data, 'sym5', level=maxlev)

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym5')

    return datarec


def hampel(vals_orig, k=7, t0=3):
    # Make copy so original not edited
    vals = pd.Series(vals_orig.copy())

    # Hampel Filter
    L = 1.4826

    rolling_median = vals.rolling(k).median()
    difference = np.abs(rolling_median - vals)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median

    # print("vals: ", vals.shape)
    return vals.to_numpy()

#-----------------------------------------------------------------------------------------

DATASET_FOLDER = "../dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114

PHASE_MIN, PHASE_MAX = 3.1389, 3.1415
AMP_MIN, AMP_MAX = 0.0, 577.6582


def read_csi_data_from_csv(path_to_csv, is_five_hhz=False, antenna_pairs=4):
    """
    Read csi data(amplitude, phase) from .csv data

    :param path_to_csv: string
    :param is_five_hhz: boolean
    :param antenna_pairs: integer
    :return: (amplitudes, phases) => (np.array of shape(data len, num_of_subcarriers * antenna_pairs),
                                     np.array of shape(data len, num_of_subcarriers * antenna_pairs))
    """

    data = pd.read_csv(path_to_csv, header=None).values

    if is_five_hhz:
        subcarries_num = SUBCARRIES_NUM_FIVE_HHZ
    else:
        subcarries_num = SUBCARRIES_NUM_TWO_HHZ

    # 1 -> to skip subcarriers numbers in data
    amplitudes = data[:, subcarries_num * 1:subcarries_num * (1 + antenna_pairs)]
    phases = data[:, subcarries_num * (1 + antenna_pairs):subcarries_num * (1 + 2 * antenna_pairs)]

    return amplitudes, phases


def read_labels_from_csv(path_to_csv):
    """
    Read labels(human activities) from csv file

    :param path_to_csv: string
    :return: labels, np.array of shape(data_len, 1)
    """

    data = pd.read_csv(path_to_csv, header=None).values
    labels = data[:, 1]

    return labels


def read_all_data_from_files(paths, is_five_hhz=True, antenna_pairs=4):
    """
    Read csi and labels data from all folders in the dataset

    :return: amplitudes, phases, labels all of shape (data len, num of subcarriers)
    """

    final_amplitudes, final_phases, final_labels = np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0))

    for index, path in enumerate(paths):
        amplitudes, phases = read_csi_data_from_csv(os.path.join(path, "data.csv"), is_five_hhz, antenna_pairs)
        labels = read_labels_from_csv(os.path.join(path, "label.csv"))

        amplitudes, phases = amplitudes[:-1], phases[:-1]  # fix the bug with the last element

        final_amplitudes = np.concatenate((final_amplitudes, amplitudes))
        final_phases = np.concatenate((final_phases, phases))
        final_labels = np.concatenate((final_labels, labels))

    return final_amplitudes, final_phases, final_labels


def read_all_data(is_five_hhz=True, antenna_pairs=4):
    all_paths = []

    for index, room in enumerate(DATA_ROOMS):
        for subroom in DATA_SUBROOMS[index]:
            all_paths.append(os.path.join(DATASET_FOLDER, room, subroom))

    return read_all_data_from_files(all_paths, is_five_hhz, antenna_pairs)

#--------------------------------wifi occupancy dataset---------------------------------
class CSIDataset(Dataset):
    """CSI Dataset."""

    def __init__(self, csv_files, window_size=32, step=1):
        from sklearn import decomposition

        self.amplitudes, self.phases, self.labels = read_all_data_from_files(csv_files)

        self.amplitudes = calibrate_amplitude(self.amplitudes)

        pca = decomposition.PCA(n_components=10)

        # self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ]))

        self.amplitudes_pca = []

        data_len = self.phases.shape[0]
        for i in range(self.phases.shape[1]):
            # self.phases[:data_len, i] = dwn_noise(hampel(self.phases[:, i]))[:data_len]
            self.amplitudes[:data_len, i] = dwn_noise(hampel(self.amplitudes[:, i]))[:data_len]

        for i in range(4):
            self.amplitudes_pca.append(
                pca.fit_transform(self.amplitudes[:, i * SUBCARRIES_NUM_FIVE_HHZ:(i + 1) * SUBCARRIES_NUM_FIVE_HHZ]))
        self.amplitudes_pca = np.array(self.amplitudes_pca)
        self.amplitudes_pca = self.amplitudes_pca.reshape((self.amplitudes_pca.shape[1], self.amplitudes_pca.shape[0] * self.amplitudes_pca.shape[2]))

        self.label_keys = list(set(self.labels))
        self.class_to_idx = {
            "standing": 0,
            "walking": 1,
            "get_down": 2,
            "sitting": 3,
            "get_up": 4,
            "lying": 5,
            "no_person": 6
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.window = window_size  # 32
        if window_size == -1:
            self.window = self.labels.shape[0] - 1

        self.step = step

    def __getitem__(self, idx):
        if self.window == 0:
            return np.append(self.amplitudes[idx], self.phases[idx]), self.class_to_idx[
                self.labels[idx + self.window - 1]]

        idx = idx * self.step
        all_xs, all_ys = [], []
        # idx = idx * self.window

        for index in range(idx, idx + self.window):
            all_xs.append(np.append(self.amplitudes[index], self.amplitudes_pca[index]))
            # all_ys.append(self.class_to_idx[self.labels[index]])

        return np.array(all_xs), self.class_to_idx[self.labels[idx + self.window - 1]]
        # return np.array(all_xs), np.array(all_ys)

    def __len__(self):
        # return self.labels.shape[0] // self.window
        # return (self.labels.shape[0] - self.window
        return int((self.labels.shape[0] - self.window) // self.step) + 1
    
#-----------------------------------------------------------------------------------------

def read_ECG(base_dir="/home/ant/RAG/data/ECG/physionet.org/files/mitdb/1.0.0/", length=600000, id=106, sampfrom=0, pysical=True, channels=[0,], interval=150, sample_step=5, draw_pictures=False):
    record_path = base_dir + f"{id}"
    record = wfdb.rdrecord(record_path, sampfrom=sampfrom, sampto=length, physical=pysical, channels=channels)
    print(record.p_signal.shape)
    import matplotlib.pyplot as plt 
    # print(f"record frequency: {record.fs}")
    ventricular_signal = record.p_signal[:]
    signal_annotation = wfdb.rdann(record_path, 'atr', sampfrom=sampfrom, sampto=length)
    # print(f"""
    # symbol: {signal_annotation.symbol}, shape={len(signal_annotation.symbol)}
    # aux_note: {signal_annotation.aux_note}, shape={len(signal_annotation.aux_note)}""")
    data_dict = {
        "N_pos": [],
        "N_signals": [],
        "V_pos": [],
        "V_signals": [],   

    }
    for i, s in enumerate(signal_annotation.symbol):
        if s == "V":
            V_begin = i
            if V_begin <= 1 or V_begin >= len(signal_annotation.symbol)-2:
                continue
            V_signals = ventricular_signal[signal_annotation.sample[V_begin]-interval:signal_annotation.sample[V_begin]+interval:sample_step]
            data_dict["V_pos"].append([interval//sample_step, V_signals[interval//sample_step]])
            data_dict["V_signals"].append(V_signals)
        elif s == "N":
            N_begin = i
            if N_begin <= 1 or N_begin >= len(signal_annotation.symbol)-2:
                continue
            N_signals = ventricular_signal[signal_annotation.sample[N_begin]-interval:signal_annotation.sample[N_begin]+interval:sample_step]
            data_dict["N_pos"].append([interval//sample_step, N_signals[interval//sample_step]])
            data_dict["N_signals"].append(N_signals)

    for k, v in data_dict.items():
        print(f"{k}: {[len(v), len(v[0])]}")
    if draw_pictures:
        # 分别展示5张N和5张V的信号,将这10个信号展示在一个2行5列的大图上,注意，保持上下两行的的纵坐标相同:
        plt.figure(figsize=(20,10))
        # 开始话2*5的大图，保持每个子图的纵坐标相同
        for i in range(1, 6):
            plt.subplot(2, 5, i)
            plt.plot(data_dict["N_signals"][i])
            plt.scatter(data_dict["N_pos"][i][0], data_dict["N_pos"][i][1], marker="*")
            plt.title(f"N_{i}")
            plt.subplot(2, 5, i+5)
            plt.plot(data_dict["V_signals"][i])
            plt.scatter(data_dict["V_pos"][i][0], data_dict["V_pos"][i][1], marker="*")
            plt.title(f"V_{i}")
        plt.show()
    return data_dict
def read_machine_data(sample_step=100):
    # 读入profile_labels文件，其中每行包括五列数据，数据之间由空格隔开
    labels = np.loadtxt('/home/ant/RAG/data/machine_detect/profile.txt', dtype=int)
    labels.shape
    label_dict = {
        "Cooler condition %": labels[:, 0],
        "Valve condition %": labels[:, 1],
        "Internal pump leakage": labels[:, 2],
        "Hydraulic accumulator": labels[:, 3],
        "Stable flag": labels[:, 4]
    } # 分别是五列数据表示的含义
    # print(label_dict["Cooler condition %"][:10])
    # 读取PS1-PS6.txt的数据：
    PS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS1.txt')
    # .shape
    PS2 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS2.txt')
    PS3 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS3.txt')
    PS4 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS4.txt')
    PS5 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS5.txt')
    PS6 = np.loadtxt('/home/ant/RAG/data/machine_detect/PS6.txt')

    # 读取EPS1.txt的数据
    EPS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/EPS1.txt')
    # print(EPS1.shape)
    # EPS1[:10]
    # 读取FS1/FS2.txt的数据
    FS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/FS1.txt')
    FS2 = np.loadtxt('/home/ant/RAG/data/machine_detect/FS2.txt')
    # print(FS1.shape)
    # FS1[:10]
    # 读取TS1-4.txt的数据
    TS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS1.txt')
    TS2 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS2.txt')
    TS3 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS3.txt')
    TS4 = np.loadtxt('/home/ant/RAG/data/machine_detect/TS4.txt')
    # print(TS1.shape)
    # TS1[:10]
    # 读取VS1.txt的数据
    VS1 = np.loadtxt('/home/ant/RAG/data/machine_detect/VS1.txt')
    # print(VS1.shape)
    # VS1[:10]
    SE = np.loadtxt('/home/ant/RAG/data/machine_detect/SE.txt')
    # print(SE.shape)   
    # SE[:10]
    CE = np.loadtxt('/home/ant/RAG/data/machine_detect/CE.txt')
    # print(CE.shape)
    # CE[:10]
    CP = np.loadtxt('/home/ant/RAG/data/machine_detect/CP.txt')
    # print(CP.shape)
    # CP[:10]
    print(PS1.shape)
    data_dict = {
        "PS1": PS1[:,::sample_step],
        "PS2": PS2[:,::sample_step],
        "PS3": PS3[:,::sample_step],
        "PS4": PS4[:,::sample_step],
        "PS5": PS5[:,::sample_step],
        "PS6": PS6[:,::sample_step],
        "EPS1": EPS1[:,::sample_step],
        "FS1": FS1[:,::sample_step//10],
        "FS2": FS2[:,::sample_step//10],
        "TS1": TS1,
        "TS2": TS2,
        "TS3": TS3,
        "TS4": TS4,
        "VS1": VS1,
        "SE": SE,
        "CE": CE,
        "CP": CP
    }
    print(f"machine_data loaded")
    return data_dict, label_dict


def time_downsample(data, time_length, time_downsample):
    # 初始化存储序列的列表
    sequences = []

    # 获取数据的长度
    data_length = data.shape[0]

    # 计算数据可以划分成多少个序列
    num_sequences = data_length // time_downsample // time_length

    # 循环生成每个序列
    for i in range(num_sequences):
        start_index = i * time_downsample * time_length
        end_index = start_index + time_length * time_downsample
        sequence = data[start_index:end_index:time_downsample, :]
        sequences.append(sequence)

    return np.array(sequences)

def read_raw_csi(root="data/wifi_csi_har_dataset/room_2/1", subcarrier_dim=40, frames_num=10, frame_downsample=2):
    val_dataset = CSIDataset([
        root,
    ])
    if root == "data/wifi_csi_har_dataset/room_2/1":
        no_person1 = val_dataset.amplitudes[5200: 6200, :] #1000
        no_person2 = val_dataset.amplitudes[12260: 12560, :] #300      12260, 12569
        no_person = np.concatenate((no_person1, no_person2), axis=0)

        walking1 = val_dataset.amplitudes[40: 560, :] #540       40, 589
        walking2 = val_dataset.amplitudes[960: 1140, :] # 180      960, 1149
        walking3 = val_dataset.amplitudes[1770: 1890, :]
        walking4 = val_dataset.amplitudes[2485: 2785, :] #  300      2485, 2799
        walking5 = val_dataset.amplitudes[2850: 3230, :]  #380
        walking6 = val_dataset.amplitudes[4375: 4715, :] #340     4375, 4729
        

        walking = np.concatenate((walking1, walking2, walking3, walking4, walking5, walking6), axis=0)


        no_person_data = time_downsample(no_person, frames_num, frame_downsample)
        walking_data = time_downsample(walking, frames_num, frame_downsample)
        yes_person_data = walking_data


    if root == "data/wifi_csi_har_dataset/room_1/4":
        no_person1 = val_dataset.amplitudes[400: 1180, :] # 780 400 1189

        
        walking1 = val_dataset.amplitudes[3660: 3860, :]  # 200
        walking2 = val_dataset.amplitudes[4100: 4380, :]  # 280
        walking3 = val_dataset.amplitudes[4900: 5220, :]  # 320
        walking4 = val_dataset.amplitudes[6050: 6350, :]  # 300
        walking5 = val_dataset.amplitudes[6640: 6780, :]  # 140
        

        walking = np.concatenate((walking1, walking2, walking3, walking4, walking5), axis=0)

        no_person_data = time_downsample(no_person1, frames_num, frame_downsample)
        walking_data = time_downsample(walking, frames_num, frame_downsample)
        yes_person_data = walking_data

    if root == "data/wifi_csi_har_dataset/room_3/1":
        no_person1 = val_dataset.amplitudes[2980: 3540, :] # 560 2980 3549

        walking1 = val_dataset.amplitudes[270: 910, :]  # 640  270 919
        walking2 = val_dataset.amplitudes[1410: 2210, :]  # 800
        walking3 = val_dataset.amplitudes[2680: 2980, :]  # 300
        walking4 = val_dataset.amplitudes[3550: 3990, :]  # 440 3550 3999

        walking = np.concatenate((walking1, walking2, walking3, walking4), axis=0)

        no_person_data = time_downsample(no_person1, frames_num, frame_downsample)
        walking_data = time_downsample(walking, frames_num, frame_downsample)
        yes_person_data = walking_data


    downsample_interval = no_person_data.shape[2] // subcarrier_dim

    # 降采样数据
    no_person_data = no_person_data[:, :, ::downsample_interval]
    yes_person_data = yes_person_data[:, :, ::downsample_interval]
    return {'no_person': no_person_data, 'have_person': yes_person_data}


def read_wifi_occupancy_data(cls_num=2):
    if cls_num == 1:
        data_dict = read_raw_csi(root="data/wifi_csi_har_dataset/room_1/4", subcarrier_dim=20, frames_num=20, frame_downsample=1)
    if cls_num == 2:
        data_dict = read_raw_csi(root="data/wifi_csi_har_dataset/room_2/1", subcarrier_dim=20, frames_num=20, frame_downsample=1)
    if cls_num == 3:
        data_dict = read_raw_csi(root="data/wifi_csi_har_dataset/room_3/1", subcarrier_dim=20, frames_num=20, frame_downsample=1)
    
    return data_dict

def read_raw_data_and_preprocess_imu(sample_step: int=5, raw_data_dir: str="/home/ant/RAG/data/IMU/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/Inertial Signals/", y_train_path: str="/home/ant/RAG/data/IMU/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt"):
    """return :
    data_dict: dict[dict[list, list, list]]

    >>> data_dict[label_id]["body_acc"] = [[body_acc_x, body_acc_y, body_acc_z], ...]
    """
    id2labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
    }
    label2ids = {
        "WALKING": 1,
        "WALKING_UPSTAIRS": 2,
        "WALKING_DOWNSTAIRS": 3,
        "SITTING": 4,
        "STANDING": 5,
        "LAYING": 6
    }
    # TODO
    signal_data_paths = {
        "body_acc_x_train_path" : raw_data_dir + "body_acc_x_train.txt",
        "body_acc_y_train_path" :  raw_data_dir + "body_acc_y_train.txt",
        "body_acc_z_train_path" :  raw_data_dir + "body_acc_z_train.txt",
        "body_gyro_x_train_path" :  raw_data_dir + "body_gyro_x_train.txt",
        "body_gyro_y_train_path" :  raw_data_dir + "body_gyro_y_train.txt",
        "body_gyro_z_train_path" :  raw_data_dir +  "body_gyro_z_train.txt",
        "total_acc_x_train_path" :  raw_data_dir + "total_acc_x_train.txt",
        "total_acc_y_train_path" :  raw_data_dir + "total_acc_y_train.txt", 
        "total_acc_z_train_path" :  raw_data_dir + "total_acc_z_train.txt",
    }
    signal_data = {}
    for signal_data_path in signal_data_paths.keys():
        with open(signal_data_paths[signal_data_path], "r") as f:
            signal_data[signal_data_path[:-5]] = np.array([list(map(float, line.split())) for line in f])
    with open(y_train_path, "r") as f:
        y_train = np.array([int(line) for line in f])
    print(Counter(y_train))
    data_dict: dict[dict[list, list, list]] = {}
    # 其中有6个key，分别代表六个活动类别，每个key中有三个list，分别代表三个传感器的数据

    for label_id in label2ids.values():
        data_dict[label_id] = {"body_acc": [], "body_gyro": [], "total_acc": []}

    for i in range(len(y_train)):
        data_dict[y_train[i]]["body_acc"].append([np.around(signal_data["body_acc_x_train"][i][::sample_step], 3), np.around(signal_data["body_acc_y_train"][i][::sample_step], 3), np.around(signal_data["body_acc_z_train"][i][::sample_step], 3)])

        data_dict[y_train[i]]["body_gyro"].append([np.around(signal_data["body_gyro_x_train"][i][::sample_step], 3), np.around(signal_data["body_gyro_y_train"][i][::sample_step], 3), np.around(signal_data["body_gyro_z_train"][i][::sample_step], 3)])

        data_dict[y_train[i]]["total_acc"].append([np.around(signal_data["total_acc_x_train"][i][::sample_step], 3), np.around(signal_data["total_acc_y_train"][i][::sample_step], 3), np.around(signal_data["total_acc_z_train"][i][::sample_step], 3)])
    return data_dict, label2ids


def read_IoT_data(task_type, sample_step=100, cls_num=2):
    assert task_type in ["imu_HAR", "machine_detection", "ecg_detection", "wifi_localization", "wifi_occupancy"] and sample_step > 0
    if task_type == "imu_HAR":
        if cls_num == 2:
            data_dict, lable_dict =  read_raw_data_and_preprocess_imu()
            data_dict = filter_data_dict_with_var(data_dict, lable_dict, thred=0.5, filter_by="body_acc", print_log=False)
            return data_dict, lable_dict
        else:
            pass # TODO
    elif task_type == "machine_detection":
        return read_machine_data(sample_step)
    elif task_type == "ecg_detection":
        data_dict = read_ECG()
        return data_dict, None
    elif task_type == "wifi_localization":
        data_dict = read_wifi_localization_data()
        return data_dict, None
    elif task_type == "wifi_occupancy":
        data_dict = read_wifi_occupancy_data(cls_num)
        return data_dict, None


class NTUIoTRSSI_Dataset(Dataset):
    def __init__(self, path_dataset):
        super(NTUIoTRSSI_Dataset, self).__init__()
        self.dim_coord = 2
        self.dim_rssi = 6
        self.path = path_dataset
        self.raw_set = self.read_txt()
        self.compact_set = None

    def read_txt(self):
        eval_law = {'np.nan': None}
        data_record_raw = []
        with open(self.path, encoding='utf-8') as file_obj:
            lines_read = file_obj.readlines()
            for line_read in lines_read:
                data_raw_temp = eval(line_read, eval_law)
                data_record_raw.append(data_raw_temp)
        data_record = np.array(data_record_raw, dtype=float)
        return data_record

    def get_compact(self, method):
        if self.compact_set is None:
            rssi_mean = []
            coord_set = self.raw_set[:, :self.dim_coord]
            rssi_set = self.raw_set[:, -self.dim_rssi:]
            unique_coords = np.unique(coord_set, return_index=False, return_counts=False, axis=0)
            for coord_tmp in unique_coords:
                idx_tmp = np.where((coord_set == coord_tmp).all(-1))[0]
                if method == 'mean':
                    rssi_mean_tmp = rssi_set[idx_tmp, :].mean(axis=0)

                elif method == 'median':
                    rssi_mean_tmp = np.median(rssi_set[idx_tmp,:], axis=0)

                rssi_mean.append(rssi_mean_tmp)
            rssi_mean = np.array(rssi_mean)
            self.compact_set = np.hstack([unique_coords, rssi_mean])

    def decrease_dataset(self, loc_num):
        unique_coords = np.unique(self.raw_set[:, :self.dim_coord], axis=0)
        decreased_set = []
        interval = len(unique_coords) // loc_num
        sample_coords = unique_coords[::interval][:loc_num]
        for coord_tmp in sample_coords:
            idx_tmp = np.where((self.raw_set[:, :self.dim_coord] == coord_tmp).all(-1))[0]
            data_tmp = self.raw_set[idx_tmp]
            decreased_set.extend(data_tmp)
        return np.array(decreased_set)


    def split_train_test(self, train_ratio=0.6, val_ratio=0.4, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        self.train_set = []
        self.val_set = []
        self.test_set = []
        coord_set = self.filtered_set[:, :self.dim_coord]
        unique_coords = np.unique(coord_set, return_index=False, return_counts=False, axis=0)
        for coord_tmp in unique_coords:
            idx_tmp = np.where((coord_set == coord_tmp).all(-1))[0]
            np.random.shuffle(idx_tmp)
            train_size = int(len(idx_tmp) * train_ratio)
            train_idx = idx_tmp[:train_size]
            val_idx = idx_tmp[train_size: train_size + int(len(idx_tmp) * val_ratio)]
            test_idx = idx_tmp[train_size + int(len(idx_tmp) * val_ratio):]
            self.train_set.extend(self.filtered_set[train_idx])
            self.val_set.extend(self.filtered_set[val_idx])
            self.test_set.extend(self.filtered_set[test_idx])
        self.test_set = np.array(self.test_set)
        self.val_set = np.array(self.val_set)
        self.train_set = np.array(self.train_set)
        return {'database_position': self.train_set[:, :self.dim_coord], 'database_rssi': self.train_set[:, -self.dim_rssi:],
                'test_position': self.test_set[:, :self.dim_coord],  'test_rssi': self.test_set[:, -self.dim_rssi:],
                'val_position': self.val_set[:, :self.dim_coord],  'val_rssi': self.val_set[:, -self.dim_rssi:]}

    def filter_outliers(self, quantile_threshold=0.8):
        """
        对数据集中的异常值进行过滤

        参数：
        - quantile_threshold: 分位数阈值，默认为0.95
        """
        self.filtered_set = []
        unique_coords = np.unique(self.raw_set[:, :self.dim_coord], axis=0)
        for coord_tmp in unique_coords:
            idx_tmp = np.where((self.raw_set[:, :self.dim_coord] == coord_tmp).all(-1))[0]
            data_tmp = self.raw_set[idx_tmp]
            # 计算统计指标，这里选择中位数作为统计指标
            statistic = np.median(data_tmp[:, self.dim_coord:], axis=0)
            # 计算样本与统计指标的差值的绝对值
            deviations = np.abs(data_tmp[:, self.dim_coord:] - statistic)
            # 计算每个样本的总偏差，这里选择每个样本的总偏差的第 quantile_threshold 分位数作为阈值
            total_deviations = np.sum(deviations, axis=1)
            threshold = np.quantile(total_deviations, quantile_threshold)
            # 过滤掉偏差较大的样本
            filtered_data_tmp = data_tmp[total_deviations <= threshold]
            self.filtered_set.extend(filtered_data_tmp)
        self.filtered_set = np.array(self.filtered_set)




    def __len__(self):
        return len(self.raw_set)

    # def __getitem__(self, item):
    #     rssi_sample = self.raw_set[item]
    #     position = rssi_sample[:self.dim_coord].astype(float)
    #     rssi_array = rssi_sample[-self.dim_rssi:].astype(float)
    #
    #     position = torch.tensor(position, dtype=torch.float32)
    #     rssi_tensor = torch.tensor(rssi_array, dtype=torch.float32)
    #
    #     return rssi_tensor, position


    def __getitem__(self, item):
        rssi_sample = self.val_set[item]
        position = rssi_sample[:self.dim_coord].astype(float)
        rssi_array = rssi_sample[-self.dim_rssi:].astype(float)

        position = torch.tensor(position, dtype=torch.float32)
        rssi_tensor = torch.tensor(rssi_array, dtype=torch.float32)

        return rssi_tensor, position

def get_dataloader_ntuiotrssi(path_dataset, batch_size):
    ntuiotrssi_dataset = NTUIoTRSSI_Dataset(path_dataset)
    ntuiotrssi_dataloader = DataLoader(dataset=ntuiotrssi_dataset, batch_size=batch_size, shuffle=True)
    return ntuiotrssi_dataloader


def read_wifi_localization_data():
    ntuiot_dataset = NTUIoTRSSI_Dataset('data/wifi_localization/rssi_position_data.txt')
    ntuiot_dataset.raw_set = ntuiot_dataset.decrease_dataset(100)
    ntuiot_dataset.filter_outliers(quantile_threshold=0.1) #1213
    data_dict = ntuiot_dataset.split_train_test(train_ratio=0.5, val_ratio=0.49, random_state=10)
    return data_dict

def read_multicls_data_and_preprocess(labels, sample_step: int=50, raw_data_dir: str="/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/"):
    """return :
    data_dict: dict[dict[list, list, list]]

    >>> data_dict[label_id]["acc"] = [[acc_x, acc_y, acc_z], ...]
    """
    # TODO
    data_dict: dict[dict[list, list, list]] = {}
    # 其中有12个key，分别代表12个活动类别，每个key中有两个list，分别代表两个传感器的数据
    for label_id in id2labels.keys():
        data_dict[label_id] = {"total_acc": [], "body_gyro": []}
    for label in labels:
        exp, user, cls_id, begin, end = label
        acc_path = f"/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/acc_exp{exp:02d}_user{user:02d}.txt"
        gyr_path = f"/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/gyro_exp{exp:02d}_user{user:02d}.txt"
        acc_data = np.loadtxt(acc_path)
        gyr_data = np.loadtxt(gyr_path)
        if cls_id <= 6:
            raw_data_acc = acc_data[begin-1:end-1:sample_step]
            raw_data_gyr = gyr_data[begin-1:end-1:sample_step]
        else:
            raw_data_acc = acc_data[begin-1:end-1:sample_step//4]
            raw_data_gyr = gyr_data[begin-1:end-1:sample_step//4]
        data_dict[cls_id]["total_acc"].append([np.around(raw_data_acc[:, 0], 3), np.around(raw_data_acc[:, 1], 3), np.around(raw_data_acc[:, 2], 3)])

        data_dict[cls_id]["body_gyro"].append([np.around(raw_data_gyr[:, 0], 3), np.around(raw_data_gyr[:, 1], 3), np.around(raw_data_gyr[:, 2], 3)])
    for label_id in data_dict.keys():
        print(f"{id2labels[label_id]}: {len(data_dict[label_id]['total_acc'])}")

    return data_dict


def filter_data_dict_with_var(data_dict, label2ids,thred: float=0.5, filter_by: str="body_acc", print_log: bool=True):
    """
    param:
    过滤掉方差百分数大于/小于thred的数据
    return:
    filtered_data_dict: dict[dict[list, list, list]]
    """
    id2labels = {v:k for k, v in label2ids.items()}
    var4cls = {
        label_id: {
            "x": [], 
            "y": [], 
            "z": []
        } for label_id in label2ids.values()
    }
    for label_id in label2ids.values():
        for i in range(len(data_dict[label_id][filter_by])):
            var4cls[label_id]["x"].append(np.var(data_dict[label_id][filter_by][i][0]))
            var4cls[label_id]["y"].append(np.var(data_dict[label_id][filter_by][i][1]))
            var4cls[label_id]["z"].append(np.var(data_dict[label_id][filter_by][i][2]))
    var4cls_sorted = {
        label_id: {
            "x": [], 
            "y": [], 
            "z": []
        } for label_id in label2ids.values()
    }
    for label_id in label2ids.values():
        var4cls_sorted[label_id]["x"] = sorted(var4cls[label_id]["x"])
        var4cls_sorted[label_id]["y"] = sorted(var4cls[label_id]["y"])
        var4cls_sorted[label_id]["z"] = sorted(var4cls[label_id]["z"])
        if print_log:
            print(f"{id2labels[label_id]} {filter_by}_x var {thred*100}% data is below {var4cls_sorted[label_id]['x'][int(len(var4cls_sorted[label_id]['x'])*thred)]}")
            print(f"{id2labels[label_id]} {filter_by}_y var {thred*100}% data is below {var4cls_sorted[label_id]['y'][int(len(var4cls_sorted[label_id]['y'])*thred)]}")
            print(f"{id2labels[label_id]} {filter_by}_z var {thred*100}% data is below {var4cls_sorted[label_id]['z'][int(len(var4cls_sorted[label_id]['z'])*thred)]}")
    # 过滤掉方差百分数大于/小于thred的数据 
    data_dict_filtered = {}
    for label_id in label2ids.values():
        data_dict_filtered[label_id] = {"body_acc": [], "body_gyro": [], "total_acc": []}
        for i in range(len(data_dict[label_id][filter_by])):
            if label_id >= 4:
                if np.var(data_dict[label_id][filter_by][i][0]) < var4cls_sorted[label_id]["x"][int(len(var4cls_sorted[label_id]["x"])*thred)] and np.var(data_dict[label_id][filter_by][i][1]) < var4cls_sorted[label_id]["y"][int(len(var4cls_sorted[label_id]["y"])*thred)] and np.var(data_dict[label_id][filter_by][i][2]) < var4cls_sorted[label_id]["z"][int(len(var4cls_sorted[label_id]["z"])*thred)]:
                    data_dict_filtered[label_id]["body_acc"].append(data_dict[label_id]["body_acc"][i])
                    data_dict_filtered[label_id]["body_gyro"].append(data_dict[label_id]["body_gyro"][i])
                    data_dict_filtered[label_id]["total_acc"].append(data_dict[label_id]["total_acc"][i])
            else:
                if np.var(data_dict[label_id][filter_by][i][0]) > var4cls_sorted[label_id]["x"][int(len(var4cls_sorted[label_id]["x"])*thred)] and np.var(data_dict[label_id][filter_by][i][1]) > var4cls_sorted[label_id]["y"][int(len(var4cls_sorted[label_id]["y"])*thred)] and np.var(data_dict[label_id][filter_by][i][2]) > var4cls_sorted[label_id]["z"][int(len(var4cls_sorted[label_id]["z"])*thred)]:
                    data_dict_filtered[label_id]["body_acc"].append(data_dict[label_id]["body_acc"][i])
                    data_dict_filtered[label_id]["body_gyro"].append(data_dict[label_id]["body_gyro"][i])
                    data_dict_filtered[label_id]["total_acc"].append(data_dict[label_id]["total_acc"][i])
        if print_log:
            print(f"{id2labels[label_id]} filtered data shape: {len(data_dict_filtered[label_id][filter_by])}")
    return data_dict_filtered

