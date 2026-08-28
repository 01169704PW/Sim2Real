import asyncio
import websockets
import json
import os
from datetime import datetime
from ina219 import INA219
import board
import busio
import adafruit_mpu6050
import math
import RPi.GPIO as GPIO
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import onnxruntime as ort
import numpy as np
import time 

# Konfiguracja podsystemu zasilania (Moduł INA219)
ina = INA219(0.1, busnum=1)
ina.configure(voltage_range=ina.RANGE_32V)

# --- INICJALIZACJA SYSTEMU INERCJALNEGO (IMU) ---
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    mpu = adafruit_mpu6050.MPU6050(i2c)
    print("[INFO] Sensor inercyjny MPU6050 (I2C) poprawnie zainicjalizowany.")
except Exception as e:
    print(f"[ERROR] Błąd inicjalizacji magistrali MPU6050: {e}")
    mpu = None

# --- INICJALIZACJA SENSORYKI DOTYKOWEJ EFEKTORÓW (GPIO) ---
# Rozkład kanałów: P0_H, P0_L, P1_H, P1_L, P2_H, P2_L, L0_H, L0_L, L1_H, L1_L, L2_H, L2_L
FOOT_SENSOR_PINS = [19, 26, 6, 13, 25, 5, 4, 17, 27, 22, 23, 24]
GPIO.setmode(GPIO.BCM) 
GPIO.setwarnings(False)

for pin in FOOT_SENSOR_PINS:
    # Ustalenie polaryzacji rezystora podciągającego (Pull-Up). Aktywacja złącza wyzwala stan niski (0).
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
print(f"[INFO] Sensoryka nacisku ({len(FOOT_SENSOR_PINS)} punktów pomiarowych) aktywna.")

# --- INICJALIZACJA UKŁADU KINEMATYCZNEGO (PWM PCA9685) ---
try:
    # Moduł prawostronny (Adres standardowy 0x41)
    pca_right = PCA9685(i2c, address=0x41)
    pca_right.frequency = 50 

    # Moduł lewostronny (Adres modyfikowany lutowaniem 0x60)
    pca_left = PCA9685(i2c, address=0x60)
    pca_left.frequency = 50

    # Tworzenie tablic instancji serwomechanizmów (wymagana kalibracja długości impulsu PWM)
    servos_right = [servo.Servo(pca_right.channels[i], min_pulse=350, max_pulse=2650) for i in range(16)]
    servos_left = [servo.Servo(pca_left.channels[i], min_pulse=350, max_pulse=2650) for i in range(16)]

    print("[INFO] Układ sterowania PWM (Dwie magistrale) zainicjalizowany poprawnie.")
except Exception as e:
    print(f"[ERROR] Błąd magistrali I2C PWM: {e}")
    pca_right, pca_left = None, None

# --- WEWNĘTRZNA STRUKTURA STANU KINEMATYCZNEGO ---
robot_state = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw_rate": 0.0,
    "yaw": 0.0,
    "buttons": {pin: False for pin in FOOT_SENSOR_PINS} 
}

# --- PARAMETRY SYMULACJI BAZOWEJ ---
MAX_TARGET_VELOCITY = 0.3
REF_STRIDE_LENGTH = 0.15
MIN_FREQUENCY = 0.0
MAX_FREQUENCY = 3.0
MAX_SERVO_DEVIATION_RAD = 0.8028 

# Parametr korekcyjny kierunków obrotu (odwrócenie polaryzacji sygnału dla konkretnych serw)
MIRRORING_ARRAY = np.array([
    -1.0, -1.0, -1.0,  
    -1.0, -1.0, -1.0,  
    -1.0, -1.0, -1.0,  
    -1.0, -1.0, -1.0,  
    -1.0, -1.0, -1.0,  
    -1.0, -1.0, -1.0   
], dtype=np.float32)

# --- REJESTRY KONTROLI LOGICZNEJ ---
cmd_vx = 0.0
cmd_vy = 0.0
gait_phases = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi], dtype=np.float32) 
previous_action = np.zeros(18, dtype=np.float32)

# --- INICJALIZACJA SILNIKA INFERENCYJNEGO (ONNX Runtime) ---
try:
    print("[INFO] Uruchamianie środowiska inferencyjnego ONNX Runtime...")
    ort_session = ort.InferenceSession("siec.onnx")
    print("[INFO] Model sieci neuronowej pomyślnie załadowany.")
except Exception as e:
    print(f"[ERROR] Błąd ładowania silnika ONNX: {e}")
    ort_session = None


P00_Offset = -5.0
P10_Offset = -8.0
P20_Offset = -5.0

# --- WEKTORY KONFIGURACJI POCZĄTKOWEJ (Kąty Bazowe) ---
# Adresacja: (P/L) - Prawa/Lewa Strona | (0,1,2) - Indeks Efektora | (0,1,2) - Segment: Goleń, Udo, Biodro
STARTING_POSITIONS_RIGHT = { 
    0: 120.0 + P00_Offset,   
    1: 120.0,  
    2: 90.0,   
    3: 120.0 + P10_Offset,   
    4: 120.0,  
    5: 90.0,   
    6: 120.0 + P20_Offset,   
    7: 120.0,  
    8: 90.0,   
}

MOVING_POSITIONS_RIGHT = { 
    0: 90.0 + P00_Offset,   
    1: 60.0,  
    2: 90.0,   
    3: 90.0 + P10_Offset,   
    4: 60.0,  
    5: 90.0,   
    6: 90.0 + P20_Offset,   
    7: 60.0,  
    8: 90.0,   
}

L21_Offset = -5.0
L20_Offset = 8.0
L11_Offset = 5.0
L10_Offset = 3.0
L01_Offset = -3.0
L00_Offset = 10.0

STARTING_POSITIONS_LEFT = { 
    7: 90.0,   
    8: 60.0 + L21_Offset,   
    9: 60.0 + L20_Offset,  
    10: 90.0 ,   
    11: 60.0 + L11_Offset,   
    12: 60.0 + L10_Offset,  
    13: 90.0,   
    14: 60.0 + L01_Offset,   
    15: 60.0 + L00_Offset,  
}

MOVING_POSITIONS_LEFT = { 
    7: 90.0,   
    8: 120.0 + L21_Offset,   
    9: 90.0 + L20_Offset,  
    10: 90.0,   
    11: 120.0 + L11_Offset,   
    12: 90.0 + L10_Offset,  
    13: 90.0,   
    14: 120.0 + L01_Offset,   
    15: 90.0 + L00_Offset,  
}

# =======================================================
# OGRANICZENIA KINEMATYCZNE (Bezpieczeństwo Sprzętowe)
# =======================================================
LIMIT_HIP = 50.0
LIMIT_FEMUR = 110.0
LIMIT_TIBIA = 110.0

LIMITS_RIGHT = {}
LIMITS_LEFT = {}

def compute_safe_margins(starting_dict, limit_dict, hip_channels, femur_channels, tibia_channels):
    """Estymacja dopuszczalnych zakresów wychylenia serwomechanizmów na podstawie pozycji referencyjnej."""
    for ch, start_kat in starting_dict.items():
        if ch in hip_channels:
            limit_dict[ch] = (start_kat - LIMIT_HIP, start_kat + LIMIT_HIP)
        elif ch in femur_channels:
            limit_dict[ch] = (start_kat - LIMIT_FEMUR, start_kat + LIMIT_FEMUR)
        elif ch in tibia_channels:
            limit_dict[ch] = (start_kat - LIMIT_TIBIA, start_kat + LIMIT_TIBIA)
        else:
            limit_dict[ch] = (0.0, 180.0) 

compute_safe_margins(STARTING_POSITIONS_RIGHT, LIMITS_RIGHT, [2, 5, 8], [1, 4, 7], [0, 3, 6])
compute_safe_margins(STARTING_POSITIONS_LEFT, LIMITS_LEFT, [13, 10, 7], [14, 11, 8], [15, 12, 9])

diagnostic_active = False
save_active = False
current_log_file = None
is_testing = False

def update_phase_oscillators(dt):
    """Mechanizm Central Pattern Generator (CPG) - aktualizacja przesunięć fazowych oscylatorów w czasie."""
    global gait_phases, cmd_vx, cmd_vy
    velocity_module = np.sqrt(cmd_vx**2 + cmd_vy**2)

    if velocity_module >= 0.01:
        frequency = np.clip(velocity_module / REF_STRIDE_LENGTH, MIN_FREQUENCY, MAX_FREQUENCY)
        omega = 2.0 * np.pi * frequency
        gait_phases += omega * dt

async def run_square_trajectory_test(direction="right"):
    """Realizacja holonomicznej trajektorii referencyjnej o topologii kwadratu."""
    global cmd_vx, cmd_vy, is_testing
    
    is_testing = True
    test_speed = 0.2
    side_duration = 3.0 
    
    direction_str = "prawoskrętnej" if direction == "right" else "lewoskrętnej"
    print(f"\n[INFO] Inicjowanie rutyny kalibracyjnej (kwadrat w kierunku: {direction_str}). Zdefiniowana prędkość: {test_speed} m/s")
    
    try:
        vx_seq = [-test_speed, 0.0, test_speed, 0.0] if direction == "right" else [test_speed, 0.0, -test_speed, 0.0]
        vy_seq = [0.0, -test_speed, 0.0, test_speed] if direction == "right" else [0.0, test_speed, 0.0, -test_speed]

        for step in range(4):
            print(f"[TEST] Wektor translacji - Segment {step+1}: Vx={vx_seq[step]}, Vy={vy_seq[step]}")
            cmd_vx, cmd_vy = vx_seq[step], vy_seq[step]
            await asyncio.sleep(side_duration)
        
        print("[INFO] Trajektoria kwadratowa zrealizowana. Stabilizowanie robota.")
        
    except asyncio.CancelledError:
        print("[WARNING] Procedura testowa przerwana zdarzeniem asynchronicznym.")
    finally:
        cmd_vx, cmd_vy = 0.0, 0.0
        is_testing = False

async def run_linear_trajectory_test():
    """Weryfikacja błędu dryftu bocznego podczas ciągłego ruchu liniowego (oś X)."""
    global cmd_vx, cmd_vy, is_testing
    
    is_testing = True
    test_speed = 0.1
    duration = 10.0
    
    print(f"\n[INFO] Rozpoczęto test utrzymywania stałego wektora prędkości: {test_speed} m/s")
    
    try:
        cmd_vx, cmd_vy = 0.0, -test_speed
        await asyncio.sleep(duration)
        print("[INFO] Zakończono profilowanie ruchu liniowego.")
        
    except asyncio.CancelledError:
        print("[WARNING] Przepływ pracy naruszony. Natychmiastowe zerowanie sterowania.")
    finally:
        cmd_vx, cmd_vy = 0.0, 0.0
        is_testing = False

async def neural_control_loop():
    """Główna pętla sterowania (50Hz) synchronizująca dane wejściowe dla sieci onnx i wymuszająca wyjścia na sprzęt."""
    global previous_action
    print("[INFO] Uruchomiono podsystem sterowania behawioralnego AI.")
    
    last_time = time.time()
    
    while True:
        if ort_session is None:
            await asyncio.sleep(1)
            continue

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        update_phase_oscillators(dt)

        # -------------------------------------------------------------
        # KONSTRUKCJA WEKTORA STANÓW (MDP)
        # -------------------------------------------------------------
        euler_imu_xy = np.array([math.radians(robot_state["roll"]), math.radians(robot_state["pitch"])], dtype=np.float32)
        yaw_rad = math.radians(robot_state["yaw"])
        
        foot_contacts = np.zeros(12, dtype=np.float32)
        
        for i in range(6): 
            pin_light = FOOT_SENSOR_PINS[i * 2]       
            pin_heavy = FOOT_SENSOR_PINS[i * 2 + 1]   
            
            if robot_state["buttons"].get(pin_light, False):
                foot_contacts[i * 2] = 1.0 

            if robot_state["buttons"].get(pin_heavy, False):
                foot_contacts[i * 2 + 1] = 1.0

        normalized_cmd = np.array([cmd_vy / MAX_TARGET_VELOCITY, cmd_vx / MAX_TARGET_VELOCITY], dtype=np.float32)
        
        phases_sin = np.sin(gait_phases)
        phases_cos = np.cos(gait_phases)
        normalized_phases = np.concatenate([phases_sin, phases_cos]).astype(np.float32)

        obs = np.concatenate([
            euler_imu_xy,                             
            [np.sin(yaw_rad), np.cos(yaw_rad)],       
            [robot_state["yaw_rate"]],              
            foot_contacts,                               
            normalized_cmd,                                
            normalized_phases,                      
            previous_action                       
        ]).astype(np.float32).reshape(1, -1)

        # -------------------------------------------------------------
        # EWALUACJA POLITYKI I APLIKACJA KINEMATYKI
        # -------------------------------------------------------------
        if abs(cmd_vx) < 0.005 and abs(cmd_vy) < 0.005:
            smoothed_action = previous_action
        else:
            try:
                ai_action = ort_session.run(None, {"input": obs})[0][0]
                smoothed_action = np.clip(ai_action, -1.0, 1.0) 
            except Exception as e:
                print(f"[ERROR] Przerwanie pracy jednostki ONNX: {e}")
                smoothed_action = np.zeros(18, dtype=np.float32)

            previous_action = smoothed_action

        deviation_deg = smoothed_action * math.degrees(MAX_SERVO_DEVIATION_RAD) * MIRRORING_ARRAY

        # --- APLIKACJA MODUŁU PRAWEGO (I2C: 0x41) ---
        move_servo(1, 2,  MOVING_POSITIONS_RIGHT[2]  + (deviation_deg[0])) 
        move_servo(1, 1,  MOVING_POSITIONS_RIGHT[1]  + (deviation_deg[1])) 
        move_servo(1, 0,  MOVING_POSITIONS_RIGHT[0]  + (deviation_deg[2])) 

        move_servo(1, 5,  MOVING_POSITIONS_RIGHT[5]  + (deviation_deg[3]))
        move_servo(1, 4,  MOVING_POSITIONS_RIGHT[4]  + (deviation_deg[4]))
        move_servo(1, 3,  MOVING_POSITIONS_RIGHT[3]  + (deviation_deg[5]))

        move_servo(1, 8,  MOVING_POSITIONS_RIGHT[8]  + (deviation_deg[6]))
        move_servo(1, 7,  MOVING_POSITIONS_RIGHT[7]  + (deviation_deg[7]))
        move_servo(1, 6,  MOVING_POSITIONS_RIGHT[6]  + (deviation_deg[8]))

        # --- APLIKACJA MODUŁU LEWEGO (I2C: 0x60) ---
        move_servo(2, 13, MOVING_POSITIONS_LEFT[13] + (deviation_deg[9]))
        move_servo(2, 14, MOVING_POSITIONS_LEFT[14] + (deviation_deg[10]))
        move_servo(2, 15, MOVING_POSITIONS_LEFT[15] + (deviation_deg[11]))

        move_servo(2, 10, MOVING_POSITIONS_LEFT[10] + (deviation_deg[12]))
        move_servo(2, 11, MOVING_POSITIONS_LEFT[11] + (deviation_deg[13]))
        move_servo(2, 12, MOVING_POSITIONS_LEFT[12] + (deviation_deg[14]))

        move_servo(2, 7,  MOVING_POSITIONS_LEFT[7]  + (deviation_deg[15]))
        move_servo(2, 8,  MOVING_POSITIONS_LEFT[8]  + (deviation_deg[16]))
        move_servo(2, 9,  MOVING_POSITIONS_LEFT[9]  + (deviation_deg[17]))

        elapsed = time.time() - current_time
        sleep_time = max(0.001, 0.02 - elapsed)
        await asyncio.sleep(sleep_time)

# =======================================================
# SPRZĘGŁO MECHANICZNE (Weryfikacja barier I2C)
# =======================================================
last_alarm_time = {} 

def move_servo(module_num, channel, target_angle):
    """Zabezpieczone wywołanie sygnału PWM, filtrowane przez słowniki progowe ograniczające zniszczenia mechaniczne."""
    global last_alarm_time
    try:
        if module_num == 1:
            min_kat, max_kat = LIMITS_RIGHT.get(channel, (0.0, 180.0))
        else:
            min_kat, max_kat = LIMITS_LEFT.get(channel, (0.0, 180.0))
            
        if target_angle < min_kat or target_angle > max_kat:
            current_t = time.time()
            key = (module_num, channel)
            if current_t - last_alarm_time.get(key, 0) > 1.0:
                print(f"[WARNING] Wykryto przekroczenie obszaru roboczego (M: {module_num}, C: {channel}). "
                      f"Żądanie: {target_angle:.1f}°, Dozwolone: {min_kat:.1f}° - {max_kat:.1f}°")
                last_alarm_time[key] = current_t

        safe_angle = max(min_kat, min(max_kat, target_angle))
        safe_angle = max(0.0, min(180.0, safe_angle))
        
        if module_num == 1 and pca_right is not None:
            servos_right[channel].angle = safe_angle
        elif module_num == 2 and pca_left is not None:
            servos_left[channel].angle = safe_angle
            
    except Exception as e:
        pass

def enforce_starting_posture():
    """Inicjalizacja strukturalna. Usztywnia maszynę w bezpiecznej pozycji transportowej."""
    print("[INFO] Kalibrowanie szkieletu sprzętowego - Pozycja startowa (0.0 rad).")
    for channel, angle in STARTING_POSITIONS_RIGHT.items():
        move_servo(1, channel, angle)
    for channel, angle in STARTING_POSITIONS_LEFT.items():
        move_servo(2, channel, angle)
    print("[INFO] Pozycja zainicjowana poprawnie.")

def enforce_walking_posture():
    """Rekalibracja serw do referencyjnej wysokości roboczej."""
    print("[INFO] Przyjmowanie referencyjnej wysokości bazy...")
    for channel, angle in MOVING_POSITIONS_RIGHT.items():
        move_servo(1, channel, angle)
    for channel, angle in MOVING_POSITIONS_LEFT.items():
        move_servo(2, channel, angle)
    print("[INFO] Struktura kinematyczna zgłasza gotowość (Stan Stabilny).")

# =======================================================
# POZYSKIWANIE DANYCH Z SYSTEMÓW WBUDOWANYCH
# =======================================================
async def internal_sensor_loop():
    """Wątek dedykowany nieprzerwanemu odczytowi żyroskopów oraz czujników nacisku. Dąży do rozdzielczości 50Hz (Filtr Komplementarny)."""
    global robot_state
    print("[INFO] System akwizycji danych środowiskowych włączony (50 Hz).")
    
    while True:
        if mpu:
            try:
                ax, ay, az = mpu.acceleration
                gx, gy, gz = mpu.gyro 
                
                acc_roll_deg = math.degrees(math.atan2(ay, az))
                acc_pitch_deg = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
                
                ALPHA = 0.96
                dt = 0.02 
                
                new_roll = ALPHA * (robot_state["roll"] + math.degrees(gx) * dt) + (1.0 - ALPHA) * acc_roll_deg
                new_pitch = ALPHA * (robot_state["pitch"] + math.degrees(gy) * dt) + (1.0 - ALPHA) * acc_pitch_deg
                
                robot_state["roll"] = round(new_roll, 2)
                robot_state["pitch"] = round(new_pitch, 2)
                
                robot_state["yaw_rate"] = round(gz, 3)
                delta_yaw = robot_state["yaw_rate"] * dt
                robot_state["yaw"] += math.degrees(delta_yaw)
            except Exception as e:
                pass
                
        for pin in FOOT_SENSOR_PINS:
            robot_state["buttons"][pin] = not GPIO.input(pin)

        await asyncio.sleep(0.02)

async def telemetry_loop(websocket):
    """Zrzut metadanych poprzez sieć z protokołem WebSocket dla zewnętrznego panelu GUI (Rejestrowanie spięć logicznych i zasilania)."""
    global diagnostic_active
    while True:
        try:
            volts = ina.voltage()
            await websocket.send(json.dumps({"type": "battery", "voltage": round(volts, 2)}))
            
            if diagnostic_active:
                active_buttons = [p for p, state in robot_state["buttons"].items() if state]
                
                diag_msg = (f"V: {volts:.2f}V | "
                            f"Roll: {robot_state['roll']}° | "
                            f"Pitch: {robot_state['pitch']}° | "
                            f"YawRot: {robot_state['yaw_rate']} | "
                            f"Active GPIO: {active_buttons}")
                await websocket.send(json.dumps({"type": "diagnostic_stream", "msg": diag_msg}))
                
        except Exception as e:
            pass
            
        await asyncio.sleep(1)

async def handle_communication(websocket):
    """Główny dyspozytor pakietów w standardzie REST-podobnym. Zabezpiecza stany aplikacji przed kolizją nadpisywanych logów."""
    global diagnostic_active, save_active, current_log_file, cmd_vx, cmd_vy
    print("[INFO] Ustanowiono poprawne połączenie na węźle WebSocket.")
    
    telemetry_task = asyncio.create_task(telemetry_loop(websocket))
    
    try:
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'joystick':
                if not is_testing: 
                    x, y, speed = data['x'], data['y'], data['speed']
                    cmd_vy = np.clip(-y * speed, -MAX_TARGET_VELOCITY, MAX_TARGET_VELOCITY) 
                    cmd_vx = np.clip(x * speed, -MAX_TARGET_VELOCITY, MAX_TARGET_VELOCITY)
                    
                    if save_active and current_log_file:
                        with open(current_log_file, "a") as f:
                            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] RUCH: X={x:.2f}, Y={y:.2f}, V={speed:.2f}\n")
                else:
                    print("[WARNING] Komenda manualna odrzucona: System wykonuje zablokowaną rutynę eksperymentalną.")

            elif data['type'] == 'speed':
                print(f"[CMD] Nadpisanie górnej granicy prędkości referencyjnej: {data['value']:.2f} m/s")

            elif data['type'] == 'custom_msg':
                cmd_txt = data['content'].strip().upper() 
                print(f"[MSG] Otrzymano zdalne wywołanie: {cmd_txt}")
                
                if save_active and current_log_file:
                    with open(current_log_file, "a") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] MSG: {data['content']}\n")
                
                if diagnostic_active:
                    resp = f"[{datetime.now().strftime('%H:%M:%S')}] Acknowledged: {data['content']}"
                    await websocket.send(json.dumps({"type": "diagnostic_stream", "msg": resp}))

                if cmd_txt == "TEST KWADRATU" or cmd_txt == "TEST":
                    if not is_testing:
                        asyncio.create_task(run_square_trajectory_test("right"))
                    else:
                        print("[WARNING] Odrzucono dyspozycję TEST_KWADRATU: Oczekuję zakonczenia bieżącej operacji we/wy.")
                        
                elif cmd_txt == "TEST ODWROTNY":
                    if not is_testing:
                        asyncio.create_task(run_square_trajectory_test("left"))
                    else:
                        print("[WARNING] Odrzucono dyspozycję TEST_ODWROTNY: Oczekuję zakonczenia bieżącej operacji we/wy.")

                elif cmd_txt == "MARSZ":
                    if not is_testing:
                        asyncio.create_task(run_linear_trajectory_test())
                    else:
                        print("[WARNING] Odrzucono dyspozycję MARSZ: Oczekuję zakonczenia bieżącej operacji we/wy.")

            elif data['type'] == 'toggle_diag':
                diagnostic_active = data['state']
                status_str = "AKTYWNY" if diagnostic_active else "NIEAKTYWNY"
                print(f"[INFO] Warstwa telemetrii o wysokiej częstotliwości: {status_str}")
                
                if diagnostic_active:
                    await websocket.send(json.dumps({"type": "diagnostic_stream", "msg": "Inicjacja subskrypcji strumienia danych."}))
                else:
                    await websocket.send(json.dumps({"type": "diagnostic_stream", "msg": "Wstrzymano transmisję strumieniową."}))
                    
            elif data['type'] == 'toggle_save':
                save_active = data['state']
                if save_active:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_log_file = f"hexapod_log_{timestamp}.txt"
                    print(f"[INFO] Rozpoczęto archiwizację logistyki dyskowej, destynacja: {current_log_file}")
                else:
                    print("[INFO] Zamknięto aktywny uchwyt dyskowy (I/O).")
                    current_log_file = None

            elif data['type'] == 'calibration' and data['command'] == 'set_base_orientation':
                old_yaw = robot_state.get("yaw", 0.0)
                robot_state["yaw"] = 0.0
                
                speed_limit = data.get('current_speed_limit', 0.0)
                voltage = data.get('voltage_state', 0.0)
                
                print(f"[KALIBRACJA] Skompensowano wariancje estymacji Yaw z {old_yaw:.2f}° na układ odniesienia 0.0°.")
                print(f"[KALIBRACJA] Kontekst zewnętrzny klienta: Próg V = {speed_limit:.2f} m/s | Szyna V = {voltage} V\n")
                
                if save_active and current_log_file:
                    with open(current_log_file, "a") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] KALIBRACJA: Pomyślny wipe orientatora bezwzględnego (Zapisany OSD: {old_yaw:.2f}°)\n")
                
                if diagnostic_active:
                    resp = f"[{datetime.now().strftime('%H:%M:%S')}] Pomyślnie przeprofilowano sensor przestrzenny (Yaw offset zresetowany)."
                    await websocket.send(json.dumps({"type": "diagnostic_stream", "msg": resp}))

            elif data['type'] == 'system' and data['command'] == 'shutdown':
                print("[WARNING] Odebrano eskalowany sygnał przerwania (SHUTDOWN). Wywoływanie procedur destrukcji zasobów...")
                if pca_right: pca_right.deinit()
                if pca_left: pca_left.deinit()
                GPIO.cleanup()
                os.system("sudo shutdown -h now")
                
    except websockets.exceptions.ConnectionClosed:
        print("[INFO] Sygnał Keep-Alive porzucony. Przywracanie wartości domyślnych.")
        cmd_vx = 0.0
        cmd_vy = 0.0
    finally:
        telemetry_task.cancel()

async def main():
    print("[INFO] Inicjalizacja rdzenia sterowania...")

    enforce_starting_posture()
    await asyncio.sleep(2)
    enforce_walking_posture()
    await asyncio.sleep(2)

    asyncio.create_task(internal_sensor_loop())
    asyncio.create_task(neural_control_loop())

    async with websockets.serve(handle_communication, "0.0.0.0", 8765):
        print("[INFO] Wątek nasłuchujący przydzielony na port: 8765.")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Zgłoszono przerwanie (SIGINT). Zwalnianie zasobów sprzętowych.")
        if pca_right is not None: pca_right.deinit()
        if pca_left is not None: pca_left.deinit()
        GPIO.cleanup()
