import pybullet as p
import pybullet_data
import math
import numpy as np
import pandas as pd
from typing import List, Tuple

# ==========================================
# 1. PARAMETRY KINEMATYCZNE I OGRANICZENIA SPRZĘTOWE
# ==========================================

# Ścieżka do modelu URDF
URDF_PATH = r""

# Zmienne morfologiczne robota sześcionożnego (wartości w mm)
BODY_RADIUS = 161.63
COXA_Z_OFFSET = 46.579
COXA_LENGTH = 79.0
FEMUR_LENGTH = 88.15
TIBIA_LENGTH = 109.0

# Kąty montażu poszczególnych odnóży w układzie lokalnym bazy (w stopniach)
MOUNTING_ANGLES_DEG = [60, 0, 300, 120, 180, 240]

# Parametry referencyjne chodu
MAX_SWING_HEIGHT = 80.0                     
REFERENCE_STRIDE_LENGTH = 0.08         
MAX_GAIT_FREQUENCY = 5.0          
MAX_STRIDE_AMPLITUDE = 100.0               
MAX_TARGET_VELOCITY = 0.3

# Obliczanie wektora translacji do domyślnej pozycji efektora 
R_HOME = COXA_LENGTH + FEMUR_LENGTH * math.cos(math.radians(60)) + TIBIA_LENGTH * math.cos(math.radians(90))
Z_HOME = COXA_Z_OFFSET - FEMUR_LENGTH * math.sin(math.radians(60)) - TIBIA_LENGTH * math.sin(math.radians(90)) 

# Domyślny wektor konfiguracji przestrzeni przegubów (radiany)
DEFAULT_POSE = np.array([
    0.0,  1.047198, 0.523599,  
    0.0,  1.047198, 0.523599,  
    0.0,  1.047198, 0.523599,  
    0.0, -1.047198, -0.523599, 
    0.0, -1.047198, -0.523599, 
    0.0, -1.047198, -0.523599  
], dtype=np.float32)

# Fizyczne ograniczenie serwomechanizmów: Maksymalne dopuszczalne odchylenie (ok. 46 stopni)
MAX_SERVO_DEVIATION_RAD = 0.8028 

# ==========================================
# 2. ODWROTNA KINEMATYKA (IK) - GENERATOR TRAJEKTORII
# ==========================================
def compute_inverse_kinematics(px: float, py: float, pz: float) -> List[float]:
    """
    Analityczne obliczanie współrzędnych przegubowych (joint space) dla zadanego 
    punktu w przestrzeni pojedynczego odnóża.
    """
    theta_1 = math.atan2(py, px)
    if abs(math.sin(theta_1)) < 1e-6: 
        term_y = px - COXA_LENGTH 
    else: 
        term_y = (py / math.sin(theta_1)) - COXA_LENGTH

    # Zabezpieczenie przed przekroczeniem dziedziny funkcji arcus cosinus, 
    # minimalizujące ryzyko niestabilności numerycznej wywołanej błędami zmiennoprzecinkowymi.
    c_theta_3 = ((pz - COXA_Z_OFFSET)**2 + term_y**2 - FEMUR_LENGTH**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
    c_theta_3 = np.clip(c_theta_3, -1.0, 1.0) 
    
    s_theta_3 = -math.sqrt(1 - c_theta_3**2) 
    theta_3 = math.atan2(s_theta_3, c_theta_3)
    
    k1 = TIBIA_LENGTH * c_theta_3 + FEMUR_LENGTH
    k2 = TIBIA_LENGTH * s_theta_3
    term_z = pz - COXA_Z_OFFSET
    r_norm = math.sqrt(term_z**2 + term_y**2)
    
    if r_norm == 0: 
        return [0, 0, 0]
        
    part1 = math.atan2(term_z / r_norm, term_y / r_norm)
    part2 = math.atan2(k2 / r_norm, k1 / r_norm)
    theta_2 = part1 - part2
    
    return [theta_1, theta_2, theta_3]

def transform_global_to_local(px_g: float, py_g: float, pz_g: float, theta_0_deg: float) -> Tuple[float, float, float]:
    """
    Transformacja z globalnego układu współrzędnych bazy robota 
    do lokalnego układu odniesienia wybranego odnóża.
    """
    theta_0 = math.radians(theta_0_deg)
    base_x = BODY_RADIUS * math.cos(theta_0)
    base_y = BODY_RADIUS * math.sin(theta_0)
    
    rel_x = px_g - base_x
    rel_y = py_g - base_y
    
    local_x = rel_x * math.cos(-theta_0) - rel_y * math.sin(-theta_0)
    local_y = rel_x * math.sin(-theta_0) + rel_y * math.cos(-theta_0)
    
    return local_x, local_y, pz_g

# ==========================================
# 3. SYMULACJA I GROMADZENIE DANYCH (DATA HARVESTING)
# ==========================================
def generate_expert_trajectories():
    """
    Proces symulacji modelu analitycznego w celu wygenerowania 
    referencyjnych par (stan, akcja) dla optymalizacji strategii algorytmu uczenia ze wzmocnieniem.
    """
    # Uruchomienie instancji silnika fizycznego w trybie bezinterfejsowym
    physics_client = p.connect(p.DIRECT) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    dataset = []
    
    # Definicja siatki weryfikacyjnej (parametryzacja prędkości i kierunku)
    directions = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    velocities = np.linspace(0.0, 0.3, 30)
    
    print("[INFO] Inicjalizacja środowiska. Rozpoczęto ekstrakcję 49-wymiarowej przestrzeni stanów MDP...")
    
    for v_mag in velocities:
        for angle in directions:
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            plane_id = p.loadURDF("plane.urdf")
            p.changeDynamics(plane_id, -1, lateralFriction=1.0)
            
            robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.228], p.getQuaternionFromEuler([0, 0, 0]), flags=p.URDF_USE_SELF_COLLISION)
            
            end_effector_indices = [2, 5, 8, 11, 14, 17]
            joint_indices = [j for j in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, j)[2] == p.JOINT_REVOLUTE]
            
            # Parametryzacja sztywności i tłumienia kontaktów
            for j in range(-1, p.getNumJoints(robot_id)):
                p.changeDynamics(robot_id, j, restitution=0.0, lateralFriction=1.0, contactStiffness=1e6, contactDamping=1e5)
            
            # Dekompozycja globalnego wektora zadanej prędkości translacyjnej
            target_vx = v_mag * math.cos(angle)  
            target_vy = v_mag * math.sin(angle)  
            
            step_frequency = 0.0 if v_mag < 0.01 else min(v_mag / REFERENCE_STRIDE_LENGTH, MAX_GAIT_FREQUENCY)
            stride_x = 0.0 if v_mag < 0.01 else np.clip((target_vx / step_frequency) * 1000.0, -MAX_STRIDE_AMPLITUDE, MAX_STRIDE_AMPLITUDE)
            stride_y = 0.0 if v_mag < 0.01 else np.clip((target_vy / step_frequency) * 1000.0, -MAX_STRIDE_AMPLITUDE, MAX_STRIDE_AMPLITUDE)
            
            dt = 1.0 / 50.0 # Okres próbkowania strategii (50 Hz)
            p.setTimeStep(1.0 / 250.0) # Krokowy integrator fizyki (250 Hz)
            
            action_buffer = np.zeros(18, dtype=np.float32)
            gait_phases = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi], dtype=np.float32)
            
            # Symulacja wstępna (ustabilizowanie sił grawitacyjnych po inicjalizacji)
            for _ in range(50):
                p.stepSimulation()

            # Rejestracja początkowego kąta odchylenia (yaw) w celu sprzężenia zwrotnego orientacji
            _, base_ori = p.getBasePositionAndOrientation(robot_id)
            reference_yaw = p.getEulerFromQuaternion(base_ori)[2]

            # Ekstrakcja danych w ustalonym horyzoncie predykcji
            for step in range(25): 
                # 1. AKWIZYCJA ZMIENNYCH STANU Z SILNIKA FIZYKI
                base_pos, base_ori = p.getBasePositionAndOrientation(robot_id)
                euler_angles = p.getEulerFromQuaternion(base_ori)
                _, ang_vel = p.getBaseVelocity(robot_id)
                _, inv_ori = p.invertTransform([0, 0, 0], base_ori)
                local_ang_vel, _ = p.multiplyTransforms([0, 0, 0], inv_ori, ang_vel, [0, 0, 0, 1])
                
                foot_contact_vector = np.zeros(12, dtype=np.float32)
                for idx, foot_link in enumerate(end_effector_indices):
                    contacts = p.getContactPoints(bodyA=robot_id, bodyB=plane_id, linkIndexA=foot_link)
                    if contacts:
                        max_force = max([k[9] for k in contacts])
                        if max_force >= 1.0: foot_contact_vector[idx * 2] = 1.0
                        if max_force >= 4.0: foot_contact_vector[idx * 2 + 1] = 1.0
                
                # Estymacja uchybu orientacji oraz sprzężenie krzyżowe (cross-coupling) wektora prędkości
                yaw_error = euler_angles[2] - reference_yaw
                normalized_velocity_cmd = np.array([-target_vy / MAX_TARGET_VELOCITY, target_vx / MAX_TARGET_VELOCITY], dtype=np.float32)
                phase_sin = np.sin(gait_phases)
                phase_cos = np.cos(gait_phases)
                
                # Konstrukcja 49-wymiarowego wektora stanu struktury (MDP)
                obs_vector = np.concatenate([
                    euler_angles[0:2],                           # Kąty pochylenia i przechylenia (Pitch, Roll)
                    [np.sin(yaw_error), np.cos(yaw_error)],      # Reprezentacja błędu kątowego (Yaw)
                    [local_ang_vel[2]],                          # Prędkość obrotowa wokół osi Z (Żyroskop)
                    foot_contact_vector,                         # Sensoryka binarna punktów styku z podłożem
                    normalized_velocity_cmd,                     # Cel kinematyczny (znormalizowany wektor 2D)
                    phase_sin, phase_cos,                        # Oscylatory fazowe układu motorycznego
                    action_buffer                                # Sprzężenie zwrotne wykonanej akcji (n-1)
                ]).astype(np.float32)
                
                # 2. ROZWIĄZYWANIE ODWROTNEJ KINEMATYKI W CELU WYZNACZENIA TRAJEKTORII (EXPERT POLICY)
                target_joint_angles = np.zeros(18, dtype=np.float32)
                for i, theta_0 in enumerate(MOUNTING_ANGLES_DEG):
                    normalized_phase = (gait_phases[i] / (2 * np.pi)) % 1.0
                    
                    if normalized_phase < 0.5:
                        is_swing = True
                        phase_t = normalized_phase * 2.0
                    else:
                        is_swing = False
                        phase_t = (normalized_phase - 0.5) * 2.0

                    base_g_x = BODY_RADIUS * math.cos(math.radians(theta_0)) + R_HOME * math.cos(math.radians(theta_0))
                    base_g_y = BODY_RADIUS * math.sin(math.radians(theta_0)) + R_HOME * math.sin(math.radians(theta_0))
                    
                    if is_swing:
                        amplitude_multiplier = -0.5 * math.cos(phase_t * math.pi)
                        pz = Z_HOME + MAX_SWING_HEIGHT * (math.sin(phase_t * math.pi))**2
                    else:
                        amplitude_multiplier = 0.5 * math.cos(phase_t * math.pi)
                        pz = Z_HOME
                        
                    px = base_g_x + amplitude_multiplier * stride_x
                    py = base_g_y + amplitude_multiplier * stride_y

                    loc_x, loc_y, loc_z = transform_global_to_local(px, py, pz, theta_0)
                    angles = compute_inverse_kinematics(loc_x, loc_y, loc_z)
                    
                    base_idx = i * 3
                    if base_idx < 9: # Odnóża prawej półsfery robota
                        target_joint_angles[base_idx:base_idx+3] = [angles[0], -angles[1], -angles[2]]
                    else: # Odnóża lewej półsfery robota
                        target_joint_angles[base_idx:base_idx+3] = [angles[0], angles[1], angles[2]]

                # 3. NORMALIZACJA PRZESTRZENI AKCJI DO DOMENY [-1, 1] 
                expert_action = (target_joint_angles - DEFAULT_POSE) / MAX_SERVO_DEVIATION_RAD
                expert_action = np.clip(expert_action, -1.0, 1.0)
                
                # Archiwizacja rekordu pomiarowego
                row = {}
                for idx_obs in range(49): row[f'obs_{idx_obs}'] = obs_vector[idx_obs]
                for idx_act in range(18): row[f'act_{idx_act}'] = expert_action[idx_act]
                dataset.append(row)
                
                # 4. AKTUALIZACJA SYMULATORA (5 cykli integratora na jeden cykl strategii MDP)
                for _ in range(5):
                    for idx_joint, val in enumerate(target_joint_angles):
                        p.setJointMotorControl2(robot_id, joint_indices[idx_joint], p.POSITION_CONTROL, targetPosition=val, force=15.0)
                    p.stepSimulation()
                
                # Inkrementacja wektora układu oscylatorów
                gait_phases += (2.0 * np.pi * step_frequency) * dt
                action_buffer = expert_action
                
    p.disconnect()
    
    # Eksport macierzy danych do pliku w formacie CSV
    df = pd.DataFrame(dataset)
    df.to_csv('tarantula_expert_dataset.csv', index=False)
    print(f"[SUKCES] Wyodrębniono {len(df)} znormalizowanych wektorów ze środowiska symulacyjnego.")

if __name__ == "__main__":
    generate_expert_trajectories()
