import socket
import os
import mysql.connector
from math import sqrt
from datetime import datetime
from dotenv import load_dotenv
import sensors
import spidev
import time
import numpy as np
import sys
import requests
import json
import pymysql

load_dotenv()  # .env 파일 로드

distance_scale = 0.00135  # 거리 센서 스케일링 값

# --- 환경 변수 기반 DB 설정 ---
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306)),
}

# --- 수신 설정 ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"[Receiver] Listening on {UDP_IP}:{UDP_PORT}")

# --- 거리 계산 함수 ---
def calculate_distance(x1, y1, x2, y2):
    pixel_distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_distance = pixel_distance * distance_scale
    return real_distance

# --- DB 저장 함수 (좌표 데이터) ---
def save_to_db(avg_x, avg_y, total_distance):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO behavior_log (timestamp, x, y, distance) VALUES (%s, %s, %s, %s)"
        values = (timestamp, avg_x, avg_y, total_distance)
        cursor.execute(query, values)
        conn.commit()
        print(f"[DB] Saved: {values}")
    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# -----------DB 연결 함수(센서용) -----------
def insert_data(table, field, value):
    try:
        conn = pymysql.connect(**db_config)
        with conn.cursor() as cursor:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 동적으로 SQL 문 생성 (timestamp와 field 값 삽입)
            sql = f"INSERT INTO {table} (timestamp, {field}) VALUES (%s, %s)"
            cursor.execute(sql, (timestamp, value))
            conn.commit()
            print(f"Data inserted: {timestamp}, {field}: {value}")
    except Exception as e:
        print(f"[DB ERROR] {e}")
    finally:
        if conn:
            conn.close()

# ----------- prox 구하기 -----------
def get_prox(distances):
    global prox1, prox2, prox3
    prox1 = 1 if distances[0] <= 4 else 0
    prox2 = 1 if distances[1] <= 4 else 0
    prox3 = 1 if distances[2] <= 4 else 0

# --- 수신 루프 ---
prev_x, prev_y = None, None
total_distance = 0.0
x_list = []
y_list = []
prox1, prox2, prox3 = 0, 0, 0 
counter = 0
sensors.init_spi()
prev_weight = None
prev_water = None
prev_light_state = False  # 조도 변화 감지용
is_night = None  # 조도 상태 저장용 변수

# 센서 데이터 축적 변수들
drinking_amount = 0.0
eating_amount = 0.0
home_amount = 0.0
home_start_time = None
prev_prox1 = 0

while True:
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode()
        x_str, y_str = message.strip().split(',')
        x, y = int(x_str), int(y_str)

        print(f"[Received] x: {x}, y: {y}")

        # 센서 데이터 수집
        weight = sensors.get_weight()
        distances = sensors.get_distance()
        water = sensors.get_water_level()
        light = sensors.get_light_level()
        get_prox(distances)  # prox1, prox2, prox3 업데이트

        print(f"Weight: {weight:.2f} g")
        print(f"Distances: {distances[0]} / {distances[1]} / {distances[2]} cm")
        print(f"prox1: {prox1}, prox2: {prox2}, prox3: {prox3}")
        print(f"Water level: {water:.1f} %")
        print(f"Light level: {light:.1f} %")
        print("-----------------------------")

        # --- 조도 50% 이상이면 야간 모드 ---
        if light >= 50:
            if is_night != 1:
                try:
                    response = requests.post("https://127.0.0.1:5005/switch_camera_night", verify=False)
                    print(f"[Light Trigger] Switched to night mode → POST /switch_camera_night → {response.status_code}")
                    is_night = 1
                except Exception as e:
                    print(f"[Light Trigger Error - Night] {e}")
        else:
            if is_night != 0:
                try:
                    response = requests.post("https://127.0.0.1:5005/switch_camera_day", verify=False)
                    print(f"[Light Trigger] Switched to day mode → POST /switch_camera_day → {response.status_code}")
                    is_night = 0
                except Exception as e:
                    print(f"[Light Trigger Error - Day] {e}")

        # --- home_data 시간 계산 ---
        if prox1 == 1 and prev_prox1 == 0:
            # prox1이 0에서 1로 변화
            home_start_time = datetime.now()
            print(f"[Home] Started at {home_start_time}")
        elif prox1 == 0 and prev_prox1 == 1 and home_start_time is not None:
            # prox1이 1에서 0으로 변화
            home_end_time = datetime.now()
            duration = (home_end_time - home_start_time).total_seconds()
            home_amount += duration
            print(f"[Home] Ended, duration: {duration:.2f} seconds, total: {home_amount:.2f}")
            home_start_time = None

        prev_prox1 = prox1

        # --- eating_data 축적 ---
        if prev_weight is not None and weight != prev_weight and prox2 == 1:
            diff = round(weight - prev_weight, 2)
            if diff < 0:
                eating_amount += abs(diff)
                print(f"[Eating] Added {abs(diff)} g, total: {eating_amount:.2f} g")

        # --- drinking_data 축적 ---
        if prev_water is not None and water != prev_water and prox3 == 1:
            diff = round(water - prev_water, 2)
            if diff < 0:
                drinking_amount += abs(diff)
                print(f"[Drinking] Added {abs(diff)} %, total: {drinking_amount:.2f} %")

        prev_weight = weight
        prev_water = water

        # 거리 계산
        if prev_x is not None and prev_y is not None:
            dist = calculate_distance(prev_x, prev_y, x, y)
            total_distance += dist

        prev_x, prev_y = x, y

        # 리스트에 추가
        x_list.append(x)
        y_list.append(y)
        counter += 1

        # 10개 수신 시 DB 저장
        if counter == 10:
            avg_x = sum(x_list) / 10
            avg_y = sum(y_list) / 10
            
            # 좌표 데이터 저장
            save_to_db(avg_x, avg_y, total_distance)
            print(f"[DB] Saved: avg_x={avg_x}, avg_y={avg_y}, total_distance={total_distance:.2f}")

            # home_data가 진행 중이라면 현재까지의 시간 계산
            if home_start_time is not None:
                current_time = datetime.now()
                duration = (current_time - home_start_time).total_seconds()
                home_amount += duration
                print(f"[Home] Ongoing session, added {duration:.2f} seconds, total: {home_amount:.2f}")
                # 다음 10개 구간을 위해 시작 시간을 현재 시간으로 재설정
                home_start_time = current_time

            # 센서 데이터들 저장
            if home_amount > 0:
                insert_data("behavior_log", "home_data", home_amount)
                print(f"[DB] home_data saved: {home_amount:.2f} seconds")

            if eating_amount > 0:
                insert_data("behavior_log", "eating_data", eating_amount)
                print(f"[DB] eating_data saved: {eating_amount:.2f} g")

            if drinking_amount > 0:
                insert_data("behavior_log", "drinking_data", drinking_amount)
                print(f"[DB] drinking_data saved: {drinking_amount:.2f} %")

            # 초기화
            x_list.clear()
            y_list.clear()
            total_distance = 0.0
            counter = 0
            drinking_amount = 0.0
            eating_amount = 0.0
            home_amount = 0.0

    except Exception as e:
        print(f"[Error] {e}")
