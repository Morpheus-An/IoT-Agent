
from imports import *

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
        elif cls_num > 2:
            labels = np.loadtxt("/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/labels.txt", dtype=int)
            data_dict, lable_dict = read_multicls_data_and_preprocess(labels, sample_step=45)
            return data_dict, lable_dict
        else:
            raise ValueError("cls_num should be greater than 1")

    elif task_type == "machine_detection":
        return read_machine_data(sample_step)
    elif task_type == "ecg_detection":
        data_dict = read_ECG()
        return data_dict, None
    elif task_type == "wifi_localization":
        pass
    elif task_type == "wifi_occupancy":
        pass

def read_multicls_data_and_preprocess(labels, sample_step: int=50, raw_data_dir: str="/home/ant/RAG/data/IMU/smartphone+based+recognition+of+human+activities+and+postural+transitions/RawData/"):
    """return :
    data_dict: dict[dict[list, list, list]]

    >>> data_dict[label_id]["acc"] = [[acc_x, acc_y, acc_z], ...]
    """
    id2labels = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
    7: "STAND_TO_SIT",
    8: "SIT_TO_STAND",
    9: "SIT_TO_LIE",
    10: "LIE_TO_SIT",
    11: "STAND_TO_LIE",
    12: "LIE_TO_STAND"}

    label2ids = {
    "WALKING": 1,
    "WALKING_UPSTAIRS": 2,
    "WALKING_DOWNSTAIRS": 3,
    "SITTING": 4,
    "STANDING": 5,
    "LAYING": 6,
    "STAND_TO_SIT": 7,
    "SIT_TO_STAND": 8,
    "SIT_TO_LIE": 9,
    "LIE_TO_SIT": 10,
    "STAND_TO_LIE": 11,
    "LIE_TO_STAND": 12
}
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

    return data_dict, label2ids


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

