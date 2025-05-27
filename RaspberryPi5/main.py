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
    return round(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 4)

# --- DB 저장 함수 ---
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
    if distances[0] <= 4:
            prox1 = 1
    else:
        prox1 = 0
    if distances[1] <= 4:
        prox2 = 1
    else:
        prox2 = 0
    if distances[2] <= 4:
        prox3 = 1
    else:
        prox3 = 0

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


        print(f"무게: {weight:.2f} g")
        print(f"거리: {distances[0]} / {distances[1]} / {distances[2]} cm")
        print(f"prox1: {prox1}, prox2: {prox2}, prox3: {prox3}")
        print(f"수위: {water:.1f} %")
        print(f"조도: {light:.1f} %")
        print("-----------------------------")

                
        # 조도 50% 이상 → 카메라 전환 POST 요청
        if light >= 50 and not prev_light_state:
            try:
                response = requests.post("http://127.0.0.1:5005/switch_camera")
                print(f"[조도 Trigger] POST /switch_camera → {response.status_code}")
                prev_light_state = True
            except Exception as e:
                print(f"[조도 Trigger Error] {e}")
        elif light < 50:
            prev_light_state = False  # 조도 다시 낮아지면 트리거 재활성화

        # prox1이 감지되면 home_data에 저장
        if prox1 == 1:
            insert_data("behavior_log", "home_data", 1)
            print("[DB] home_data 기록 완료")

        # 무게 변화 시 + prox2가 1일 때 eating_data 저장
        if prev_weight is not None and weight != prev_weight and prox2 == 1:
            diff = round(weight - prev_weight, 2)
            insert_data("behavior_log", "eating_data", diff)
            print(f"[DB] eating_data 기록 완료: {diff}g")

        # 수위 변화 시 + prox3가 1일 때 drinking_data 저장
        if prev_water is not None and water != prev_water and prox3 == 1:
            diff = round(water - prev_water, 2)
            insert_data("behavior_log", "drinking_data", diff)
            print(f"[DB] drinking_data 기록 완료: {diff}%")
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
            save_to_db(avg_x, avg_y, total_distance)

            # 초기화
            x_list.clear()
            y_list.clear()
            total_distance = 0.0
            counter = 0

    except Exception as e:
        print(f"[Error] {e}")
