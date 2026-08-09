import torch
from stable_baselines3 import PPO

print("[INFO] Inicjalizacja skryptu eksportu modelu polityki do środowiska uruchomieniowego (ONNX)...")

# =====================================================================
# 1. HERMETYZACJA POLITYKI DO EKSPORTU (ONNXABLE WRAPPER)
# =====================================================================
class OnnxablePolicy(torch.nn.Module):
    """
    Nakładka strukturalna dla obiektu Actor-Critic. 
    Izoluje sieć neuronową Aktora od komponentów uczenia, umożliwiając 
    bezpośrednią propagację w przód (forward pass) wymaganą przez kompilator ONNX.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        # 1. Ekstrakcja nieliniowych cech z surowego wektora stanu
        features = self.policy.extract_features(observation)
        
        # 2. Przetwarzanie przez warstwy gęste polityki (Actor Network)
        # Drugi element zwracanej krotki (latent_vf) jest ignorowany (dotyczy Krytyka)
        latent_pi, _ = self.policy.mlp_extractor(features)
        
        # 3. Propagacja przez warstwę wyjściową decyzyjną
        # Generowanie zdeterminowanego wektora akcji (mean_actions) bez rozkładu Gaussa
        return self.policy.action_net(latent_pi)

# =====================================================================
# 2. ŁADOWANIE ZAPISANEGO MODELU (CHECKPOINT)
# =====================================================================
print("[INFO] Wczytywanie zoptymalizowanych wag modelu z dysku...")

# UWAGA: Ścieżka do archiwum .zip musi zostać zaktualizowana na środowisku docelowym!
try:
    model = PPO.load("OSTATECZNY_NAJLEPSZY_MOZG", device="cpu")
    print("[INFO] Pomyślnie załadowano architekturę PPO.")
except Exception as e:
    print(f"[ERROR] Błąd dostępu do wczytywanego archiwum: {e}")
    exit(1)

# Izolacja polityki i zamrożenie wag (tryb inferencji)
onnxable_model = OnnxablePolicy(model.policy)
onnxable_model.eval()

# =====================================================================
# 3. GENEROWANIE TENSORA TESTOWEGO (DUMMY INPUT)
# =====================================================================
# Tensory testowe weryfikują zgodność wymiarową ścieżki przejścia we/wy
# Aktualna definicja MDP zakłada Batch_Size=1, Observation_Space=49
try:
    dummy_observation = torch.randn(1, 49)
    print("[INFO] Pomyślnie wygenerowano wzorcowy tensor wejściowy [1, 49].")
except Exception as e:
    print(f"[ERROR] Konflikt alokacji tensora wejściowego: {e}")
    exit(1)

# =====================================================================
# 4. EKSPORT ŚRODOWISKA DO PLIKU WYKONAWCZEGO (.onnx)
# =====================================================================
print("[INFO] Kompilowanie reprezentacji grafowej sieci neuronowej...")

try:
    torch.onnx.export(
        onnxable_model,              
        dummy_observation,           
        "siec.onnx",       
        export_params=True,          
        opset_version=18,            
        input_names=["input"],       
        output_names=["output"]
    )
    print("[INFO] Procedura zakończona sukcesem.")
    print("[INFO] Graf wykonawczy został zaprogramowany w pliku 'siec.onnx'.")
except Exception as e:
    print(f"[ERROR] Błąd kompilatora ONNX: {e}")