# -4-42-
# 遍历4个机理，并遍历42组初始条件（成功）

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd


def detect_ignition(T_prev, T_current, threshold):
    """
    检测是否发生点火。
    参数:
        T_prev: 上一个时间点的温度
        T_current: 当前时间点的温度
        threshold: 触发点火的温度上升阈值
    返回:
        点火发生返回 True，否则返回 False。
    """
    return T_current - T_prev > threshold

def adjust_timestep(ignition_detected, dt_initial, dt_min, dt_max):
    """
    根据点火状态调整时间步长。
    参数:
        ignition_detected: 是否检测到点火
        dt_initial: 初始时间步长
        dt_min: 最小时间步长
        dt_max: 最大时间步长
    返回:
        调整后的时间步长。
    """
    if ignition_detected:
        return max(dt_initial, dt_min)
    else:
        return min(dt_initial, dt_max)

# 读取Excel文件
file_path = 'E:/Desktop/Postgraduate/mech/simplify/initial_conditions.xlsx'
df = pd.read_excel(file_path)

# 初始化用于存储初始条件的列表
initial_conditions_list = []

# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    T = row['T']
    P = row['P']
    # 从各列中读取组分的摩尔比
    H2 = row['H2']
    NH3 = row['NH3']
    O2 = row['O2']
    N2 = row['N2']
    Ar = row['Ar']

    # 构造组分字符串
    composition = f'H2:{H2}, NH3:{NH3}, O2:{O2}, N2:{N2}, Ar:{Ar}'

    # 将当前行的数据保存为一个字典
    condition = {
        'Temperature': T,
        'Pressure': P,
        'Composition': composition
    }

    # 添加到列表中
    initial_conditions_list.append(condition)

# 设置Cantera模拟和初始条件
#initial_conditions = [
#    {'T': 945.06, 'P': 10.03, 'composition': 'NH3:9, H2:1, O2:7.25, N2:3.4, Ar:23.85'},
#    {'T': 964.6, 'P': 9.94, 'composition': 'NH3:9, H2:1, O2:7.25, N2:3.4, Ar:23.85'}
#]

# 设置需要遍历的根目录
mech_folder = 'E:/Desktop/Postgraduate/mech/simplify/'

# 使用glob模块查找所有mech.yaml文件
mech_files = glob.glob(os.path.join(mech_folder, '**', '*.yaml'), recursive=True)

# 初始化matplotlib的图形和轴对象
fig, ax = plt.subplots(figsize=(10, 6))

# 创建一个Excel writer
with pd.ExcelWriter(os.path.join(mech_folder, 'simulation_results.xlsx')) as writer:
    for mech_path in mech_files:
        Mech = ct.Solution(mech_path)
        # 为每个机理文件创建一个DateFrame来存储结果
        results_df = pd.DataFrame(columns=['Temperature', 'Pressure', 'Composition', 'Ignition Delay Time'])
        for idx, condition in enumerate(initial_conditions_list):
            Mech.TPX = condition['Temperature'], condition['Pressure'] * ct.one_atm, condition['Composition']
            reactor = ct.IdealGasReactor(Mech)
            reactor_network = ct.ReactorNet([reactor])

            # 时间参数
            t_end = 150 / 1000  # 模拟结束时间，单位转换为秒
            dt_initial = 1e-6  # 初始时间步长，单位秒
            dt_max = 1e-5  # 最大时间步长，单位秒
            dt_min = 1e-7  # 最小时间步长，单位秒
            ignition_threshold = 50  # 点火检测温度上升阈值，单位K

            time = 0.0
            T_prev = condition['Temperature']
            ignition_detected = False
            times = []
            temperatures = []

            while time < t_end:
                # 动态调整时间步长
                dt = adjust_timestep(ignition_detected, dt_initial, dt_min, dt_max)
                # 推进反应
                reactor_network.advance(time + dt)
                T_current = reactor.T
                time = reactor_network.time

                # 检测点火
                if not ignition_detected and detect_ignition(T_prev, T_current, ignition_threshold):
                    ignition_detected = True
                    print(f"Ignition detected at {time * 1000:.4f} ms, adjusting timestep.")
                    ignition_time_ms = time * 1000  # 记录点火时间，转换为毫秒
                # 保存状态数据
                times.append(time * 1000)  # 时间单位转换为毫秒
                temperatures.append(T_current)
                T_prev = T_current

            if not ignition_detected:
                ignition_time_ms = "False"

            # 将点火时间和相关信息添加到列表中
            results_df.loc[idx] = [condition['Temperature'], condition['Pressure'], condition['Composition'], ignition_time_ms if ignition_detected else 'None']
        #将结果保存到Excel的不同sheet中
        mech_basename = os.path.basename(mech_path).split('.')[0]
        results_df.to_excel(writer, sheet_name=mech_basename, index=False)

print(f'simulation Done')
