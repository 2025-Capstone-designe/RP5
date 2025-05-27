import spidev
import time
import numpy as np
import os
import sys
spi = None

def init_spi():
    global spi
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1350000  # 가장 높은 속도에 맞춰 설정

def close_spi():
    if spi:
        spi.close()

def read_adc(channel):
    if not 0 <= channel <= 7:
        raise ValueError("채널은 0~7 사이여야 합니다.")
    r = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((r[1] & 3) << 8) + r[2]

def adc_to_voltage(adc_value, vref=3.3):
    return (adc_value / 1023.0) * vref

# ---------------- 무게 센서 ----------------
# 캘리브레이션된 데이터
weights_g = np.array([0, 100, 200, 300, 400, 500])        # g 단위
adc_values = np.array([20, 150, 270, 370, 450, 520])      # FSR ADC 측정값
a, b, c = np.polyfit(adc_values, weights_g, 2)

def adc_to_weight(adc_value):
    return max(0, a * adc_value**2 + b * adc_value + c)

def get_weight():
    adc_val = read_adc(0)  # CH0
    return round(adc_to_weight(adc_val), 2)

# ---------------- 거리 센서 ----------------
def voltage_to_distance_cm(voltage):
    if voltage < 0.25:
        return 15.0
    return min(15.0, max(2.0, (27.86 * (voltage ** -1.15)) / 10.0))

def get_distance():
    channels = [1, 2, 3]
    distances = []
    for ch in channels:
        adc_val = read_adc(ch)
        voltage = adc_to_voltage(adc_val)
        dist = voltage_to_distance_cm(voltage)
        distances.append(round(dist, 2))
    return distances  # [거리1, 거리2, 거리3]

# ---------------- 수위 센서 ----------------
def read_water_level_percent(channel=4, vref=3.3):
    adc_val = read_adc(channel)
    voltage = adc_to_voltage(adc_val, vref)

    min_v = 0.3
    max_v = 2.7

    if voltage < min_v:
        percent = 0.0
    elif voltage > max_v:
        percent = 100.0
    else:
        percent = (voltage - min_v) / (max_v - min_v) * 100.0

    return round(percent, 1)

def get_water_level():
    return read_water_level_percent(4)

# ---------------- 카메라 센서 ----------------
def takePhoto(cameraId, filename):
    cmd  = f"libcamera-still -n -t 1000 --camera {cameraId} -o {filename}"
    print(f"Capturing from camera {cameraId}")
    os.system(cmd)

# ---------------- 조도 센서 ----------------
def get_light_level(channel=5, vref=3.3):
    adc_val = read_adc(channel)
    voltage = adc_to_voltage(adc_val, vref)

    # 조도 값이 낮으면 어둡고, 높으면 밝음
    # ADC 값 또는 전압 그대로 반환하거나, 비율(%)로 환산할 수 있음
    percent = min(100.0, max(0.0, (voltage / vref) * 100.0))
    
    return round(percent, 1)  # % 단위로 반환

# ---------------- 테스트 ----------------
if __name__ == "__main__":
    try:
        init_spi()
        while True:
            weight = get_weight()
            distances = get_distance()
            water = get_water_level()
            light = get_light_level()

            print(f"무게: {weight:.2f} g")
            print(f"거리: {distances[0]} / {distances[1]} / {distances[2]} cm")
            print(f"수위: {water:.1f} %")
            print(f"조도: {light:.1f} %")
            print("-----------------------------")
            time.sleep(1)
    except KeyboardInterrupt:
        print("측정 종료")
    finally:
        close_spi()


