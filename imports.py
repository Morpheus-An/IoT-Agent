from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
import wikipedia
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
import os 
from haystack import Pipeline
import numpy as np 
import torch
from openai import OpenAI 
from collections import Counter 
from matplotlib import pyplot as plt 
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from typing import Any, Dict, List, Optional, Union
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.utils.device import ComponentDevice
from haystack_integrations.components.generators.anthropic import AnthropicGenerator
import time 
from openAI_API_key import *
import pdb
import wfdb
import datetime


EMBEDDER_MODEL = "thenlper/gte-large"
EMBEDDER_MODEL_LOCAL = "/home/nfs02/ant/thenlper-gte-large"
RANKER_MODEL = "BAAI/bge-reranker-base"
RANKER_MODEL_LOCAL = "/home/nfs02/ant/baai-bge-reranker-base"
MODEL = {
        "gpt3.5": "gpt-3.5-turbo", 
        "gpt4": "gpt-4-turbo-preview",
        "llama2": "/home/nfs02/ant/LLaMa2-7b-32k",
        "Mistral": "/home/ant/RAG/models/Mistral-7b-instruct-v0.3",
}
# hoices=["imu_HAR", "machine_detection", "ecg_detection", "wifi_localization", "wifi_occupancy"],
content4retrieve_domain = {
    "machine_detection": """
The data set was experimentally obtained with a hydraulic test rig. This test rig consists of a primary working and a secondary cooling-filtration circuit which are connected via the oil tank [1], [2]. The system cyclically repeats constant load cycles (duration 60 seconds) and measures process values such as pressures, volume flows and temperatures while the condition of four hydraulic components (cooler, valve, pump and accumulator) is quantitatively varied.
    
Attributes are sensor data (all numeric and continuous) from measurements taken at the same point in time, respectively, of a hydraulic test rig's working cycle.

Temepurature sensors (TS) measure the temperature of the oil at different points in the hydraulic system. The sensors are named TS1, TS2, TS3, and TS4. The temperature is measured in degrees Celsius.
Efficiency factor sensors (SE) measure the efficiency of the cooler. The efficiency is calculated from the ratio of the actual cooling power to the ideal cooling power. The sensors are named SE. The efficiency factor is given in percentage.
Cooling power sensors (CP) measure the cooling power in kilowatts. The sensors are named CP1 and CP2. The cooling power is measured in kilowatts.

For each sensor, we collected 60 data points over a period of 60 seconds at a monitoring frequency of 1Hz (measuring sensor data once every second), forming a time series of length 60. We measured the following sequences using temperature sensors, Cooling power sensors, and Cooling efficiency sensors:

1. **Temperature Change Sequence**: Reflects the machine's temperature variation over 60 seconds, in degrees Celsius. By analyzing this sequence, you can assess whether the cooling equipment is operating normally. Typically, when the cooling system is functioning well, the machine's temperature is relatively low and does not fluctuate too significantly. If the temperature consistently remains at a high degrees Celsius or fluctuates significantly, it may indicate an abnormal issue with the cooling equipment.

2. **Cooling Power Change Sequence**: Reflects the variation in the cooling power of the machine's cooling equipment over 60 seconds, in kilowatts (KW). By analyzing this sequence, you can determine if the cooling equipment is operating normally. Generally, when the cooling system is functioning properly, the cooling power is relatively high and remains relatively stable throughout the period. If the power consistently stays low, it may suggest an abnormal issue with the cooling equipment.

3. **Cooling Efficiency Change Sequence**: Reflects the variation in the efficiency of the machine's cooling equipment over 60 seconds, in percentage (%). By analyzing this sequence, you can judge if the cooling equipment is operating normally. Typically, when the cooling system is working well, the cooling efficiency is relatively high, otherwise, it indicates that there may be an abnormal issue with the cooling equipment.

Please analyze the data step by step to explain what it reflects, and then provide your final answer based on your analysis: "Is the machine's cooling system functioning properly?"
""",
    "imu_HAR": {
        "2cls" : """
1. Triaxial acceleration signal: 
The provided three-axis acceleration signals contain acceleration data for the X-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting of 26 data samples, measured at a fixed time interval with a frequency of 10Hz(10 samples is collected per second). The unit is gravitational acceleration (g), equivalent to 9.8m/s^2. It's important to note that the measured acceleration is influenced by gravity, meaning the acceleration measurement along a certain axis will be affected by the vertical downward force of gravity. 
2. Triaxial angular velocity signal: 
The provided three-axis angular velocity signals contain angular velocity data for the X-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting of 26 data samples, measured at a fixed time interval with a frequency of 10Hz. The unit is radians per second (rad/s).

You need to comprehensively analyze the acceleration and angular velocity data on each axis. For each axis, you should analyze not only the magnitude and direction of each sampled data (the direction is determined by the positive or negative sign in the data) but also the changes and fluctuations in the sequential data along that axis. This analysis helps in understanding the subject's motion status. For example, signals with greater fluctuations in sample data in the sequence often indicate the subject is engaging in more vigorous activities like WALKING, whereas signals with smaller fluctuations in sample data often indicate the subject is engaged in calmer activities like STANDING.""",
        "mcls": """
1. Triaxial acceleration signal: 
The provided three-axis acceleration signals contain acceleration data for the X-axisY-axis, and Z-axis respectively. Each axis's data is a time-series signal consisting omultiple data samples, measured at a fixed time interval with a frequency of 10Hz(1samples is collected per second). The unit is gravitational acceleration (g), equivalent t9.8m/s^2. It's important to note that the measured acceleration is influenced by gravitymeaning the acceleration measurement along a certain axis will be affected by the verticadownward force of gravity. 
2. Triaxial angular velocity signal: 
The provided three-axis angular velocity signals contain angular velocity data for thX-axis, Y-axis, and Z-axis respectively. Each axis's data is a time-series signaconsisting of multiple data samples, measured at a fixed time interval with a frequency o10Hz. The unit is radians per second (rad/s)

You need to comprehensively analyze the acceleration and angular velocity data on eacaxis. For each axis, you should analyze not only the magnitude and direction of eacsampled data (the direction is determined by the positive or negative sign in the data) bualso the changes and fluctuations in the sequential data along that axis. This analysihelps in understanding the subject's motion status. 
For example, when the signal consistently shows significant fluctuations, it indicates tha the person may be engaged in continuous activities, such as WALKING_UPSTAIRS. On the other hand, when the signal consistently displays fewer fluctuations, it suggests that the person mabe in a relatively calm state, such as LAYING. However, if there are differing pattern between segments of the signal sequence, and there are notable changes, particularly on certain axes during specific periods, it suggests that the person may be transitioning between activity states, such as in the case of LIE-TO-SIT"""
},
    "ecg_detection": """
The ECG data is collected from a patient's heart. The data consists of a series of electrical signals that represent the heart's electrical activity. The signals are measured in millivolts (mV) and are recorded over a period of time at the sampling frequency of 60Hz. This means there is an interval of 0.017 seconds between the two voltage values.  The data is divided into two categories: normal heartbeats (N) and ventricular ectopic beats (V). The normal heartbeats represent the regular electrical activity of the heart, while the ventricular ectopic beats represent abnormal electrical activity. The data is collected using a single-channel ECG device.  Normal heartbeat (N) signals are characterized by a consistent pattern of electrical activity, while premature ventricular contraction (V) signals exhibit irregular patterns that deviate from the normal rhythm. The ECG data provides valuable insights into the patient's cardiac health and can help in diagnosing various heart conditions.  
Please analyze the data step by step to explain what it reflects, and then provide your final answer based on your analysis: "Is it a Normal heartbeat(N) or Premature ventricular contraction beat(V)?"
""",
    "wifi_localization": "",
    "wifi_occupancy": ""
}
Role_des = {
    "machine_detection": """
As a seasoned machine evaluation expert with a profound understanding of hydraulic systems, you possess the following key abilities and knowledge:

- **System Comprehension**: You are well-versed in the inner workings of hydraulic systems, including the functions of key components such as coolers, pumps, valves, and accumulators, as well as their interplay.
- **Data Analysis**: You are adept at handling and analyzing complex datasets, employing advanced techniques like time series analysis, signal processing, and pattern recognition to uncover the underlying stories in the data.
- **Sensor Interpretation**: You have a deep understanding of various sensor data (e.g., pressure, temperature, flow, vibration) and can discern the operational status of machinery that these data represent.
- **Fault Diagnosis**: You are familiar with the types of malfunctions that can occur in hydraulic systems and can recognize the signs of these faults in sensor data, enabling you to monitor the machine's operating condition and diagnose potential issues.
- **System Synergy**: You understand how the different parts of a machine work together and how anomalies in one component can trigger a cascade effect throughout the system.

In the upcoming task of machine condition assessment, you will leverage your expertise and skills to conduct an in-depth analysis of the data collected by the hydraulic test rig. You will evaluate the operational status of the cooler and other critical components, providing professional insights and recommendations for machine performance optimization and preventive maintenance.""",
    "imu_HAR": """
You are an assistant sports scientist, specialized in analyzing sensor data to understand human movement and activity patterns. Your expertise in interpreting accelerometer sensor data makes you an expert in human activity recognition tasks. Your role is to assist users in determining the status of human activities by analyzing accelerometer data.
Your training enables you to interpret and analyze the data collected by accelerometer sensors, thereby identifying different motion patterns. You understand the acceleration patterns generated by the human body in various activities and can determine the current activity status based on changes in the data.
Your professional knowledge includes, but is not limited to:
1. Human Biomechanics: You understand the acceleration patterns generated by the human body in different activity modes and their relationship with specific activities.
2. Data Analysis and Pattern Recognition: You can utilize machine learning and pattern recognition techniques to analyze and process sensor data, accurately identifying human activities.
3. Exercise Physiology: You understand the physiological changes that occur in the human body during exercise, which can assist in activity recognition.
As an assistant sports scientist, your task is to classify human activities based on the acceleration data you receive, helping users better understand and monitor their exercise activities.""",
    "ecg_detection": """
You are an experienced physician who is familiar with various types of electrocardiogram (ECG) data. You can easily make preliminary judgments on whether heartbeats are abnormal based on ECG data. You possess the following medical and domain knowledge:

1. **ECG Interpretation:** You understand the basic principles of electrocardiography and know how to interpret ECG waveforms, including identifying different phases of the cardiac cycle and recognizing abnormalities.

2. **Cardiac Physiology:** You are familiar with the physiological functions of the heart, the generation and propagation of cardiac electrical signals, and the characteristics and manifestations of various cardiac arrhythmias.

3. **Recognition of ECG Abnormalities:** You are able to identify abnormal waveforms in ECG data, such as arrhythmias, myocardial ischemia, myocardial infarction, etc., and differentiate them from normal ECG patterns.

4. **Medical Statistics:** You are proficient in statistical analysis of ECG data, identification of outliers, and quantitative assessment of abnormalities.

5. **Clinical Experience:** You have extensive clinical experience to integrate ECG data with patient symptoms and medical history for accurate diagnosis and evaluation.

6. **Medical Ethics and Legal Knowledge:** You understant medical ethics and legal regulations to ensure confidentiality and lawful use of patient data.

The combined application of these domain knowledge and skills would enable you to accurately assess whether there are any abnormalities in the ECG data and provide relevant analysis and interpretation.
""",
    "wifi_localization":"",
    "wifi_occupancy":""
}
# id2labels = {
#     1: "WALKING",
#     2: "WALKING_UPSTAIRS",
#     3: "WALKING_DOWNSTAIRS",
#     4: "SITTING",
#     5: "STANDING",
#     6: "LAYING"
# }
# id2labels = {
#     1: "WALKING",
#     2: "WALKING_UPSTAIRS",
#     3: "WALKING_DOWNSTAIRS",
#     4: "SITTING",
#     5: "STANDING",
#     6: "LAYING",
#     7: "STAND_TO_SIT",
#     8: "SIT_TO_STAND",
#     9: "SIT_TO_LIE",
#     10: "LIE_TO_SIT",
#     11: "STAND_TO_LIE",
#     12: "LIE_TO_STAND"}

# label2ids = {
#     "WALKING": 1,
#     "WALKING_UPSTAIRS": 2,
#     "WALKING_DOWNSTAIRS": 3,
#     "SITTING": 4,
#     "STANDING": 5,
#     "LAYING": 6,
#     "STAND_TO_SIT": 7,
#     "SIT_TO_STAND": 8,
#     "SIT_TO_LIE": 9,
#     "LIE_TO_SIT": 10,
#     "STAND_TO_LIE": 11,
#     "LIE_TO_STAND": 12
# }
# label2ids = {
#     "WALKING": 1,
#     "WALKING_UPSTAIRS": 2,
#     "WALKING_DOWNSTAIRS": 3,
#     "SITTING": 4,
#     "STANDING": 5,
#     "LAYING": 6
# }


