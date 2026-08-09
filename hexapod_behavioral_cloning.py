import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

print("[INFO] Inicjalizacja środowiska uczenia nadzorowanego (Behavioral Cloning)...")

# ==========================================
# 1. STRUKTURA DANYCH REFERENCYJNYCH
# ==========================================
class ExpertTrajectoryDataset(Dataset):
    """
    Zbiór danych integrujący obserwacje środowiskowe i referencyjne akcje 
    wygenerowane przez politykę ekspercką (model analityczny).
    """
    def __init__(self, csv_file: str):
        df = pd.read_csv(csv_file)
        
        # Ekstrakcja 49 wymiarów wektora stanu i 18 wymiarów przestrzeni akcji
        obs_cols = [f'obs_{i}' for i in range(49)]
        act_cols = [f'act_{i}' for i in range(18)]
        
        self.obs = torch.tensor(df[obs_cols].values, dtype=torch.float32)
        self.act = torch.tensor(df[act_cols].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return self.obs[idx], self.act[idx]

# ==========================================
# 2. ŚRODOWISKO ZAŚLEPKOWE (DUMMY ENV)
# ==========================================
class DummyHexapodEnv(gym.Env):
    """
    Minimalne środowisko implementujące interfejs Gymnasium.
    Wymagane przez bibliotekę Stable Baselines3 do prawidłowej 
    inicjalizacji wymiarów przestrzeni akcji i obserwacji modelu PPO.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(49,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)

    def step(self, action): 
        return np.zeros(49), 0, False, False, {}

    def reset(self, seed=None, options=None): 
        return np.zeros(49), {}

# ==========================================
# 3. PROCEDURA KLONOWANIA BEHAWIORALNEGO
# ==========================================
def perform_behavioral_cloning():
    env = DummyHexapodEnv()
    
    # Architektura współbieżna z docelowym środowiskiem uczenia ze wzmocnieniem
    policy_architecture = dict(
        activation_fn=nn.ReLU, 
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])
    )
    
    print("[INFO] Alokacja struktury sieci neuronowej PPO...")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_architecture, verbose=1)
    
    # Inicjalizacja strumienia danych z modelem eksperckim
    dataset = ExpertTrajectoryDataset('tarantula_expert_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Optymalizacja sieci Aktora (Policy Network) za pomocą uczenia nadzorowanego
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    EPOCHS = 60 
    
    print("[INFO] Rozpoczęto fazę uczenia nadzorowanego (Behavioral Cloning)...")
    model.policy.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for obs_batch, act_batch in dataloader:
            obs_batch = obs_batch.to(model.device)
            act_batch = act_batch.to(model.device)
            
            # Ekstrakcja cech wejściowych (Feature Extractor z SB3)
            features = model.policy.extract_features(obs_batch)
            latent_pi, _ = model.policy.mlp_extractor(features)
            predicted_actions = model.policy.action_net(latent_pi)
            
            # Estymacja funkcji strat (MSE) pomiędzy predykcją a wzorcem eksperckim
            loss = criterion(predicted_actions, act_batch)
            
            # Wsteczna propagacja błędu (Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoka [{epoch:03d}/{EPOCHS}] | Błąd średniokwadratowy (MSE): {avg_loss:.6f}")

    output_model_name = "PPO_Pretrained_Behavioral_Cloning"
    model.save(output_model_name)
    print(f"\n[INFO] Procedura zakończona. Model zapisano jako '{output_model_name}.zip'")
    print("[INFO] Archiwum przygotowane do wdrożenia w głównej pętli PPO.")

if __name__ == "__main__":
    perform_behavioral_cloning()