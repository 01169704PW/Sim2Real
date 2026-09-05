# Sterowanie robotem kroczącym z wykorzystaniem uczenia ze wzmocnieniem i transferu Sim2Real

Oficjalne repozytorium kodu źródłowego do pracy magisterskiej:  
**„Sterowanie robotem kroczącym z wykorzystaniem uczenia ze wzmocnieniem i transferu Sim2Real na podstawie danych z czujników inercyjnych i kontaktowych”**  
**Autor:** Michał Witkowski  
**Wydział Elektryczny, Politechnika Warszawska (2026)**

---

## 📌 O projekcie

Projekt obejmuje opracowanie, optymalizację oraz wdrożenie sprzętowe (Sim2Real) hybrydowego układu sterowania ruchem sześcionożnego robota kroczącego o 18 stopniach swobody (18-DoF). 

Kluczowe cechy architektury:
- **Hybrydowy oscylator CPG + PPO:** połączenie deterministycznego oscylatora fazowego z nieliniową korekcją trajektorii generowaną przez sieć neuronową (PPO).
- **Częściowa obserwowalność (POMDP):** sterowanie bez enkoderów położenia w serwomechanizmach i bez sprzężenia siłowego — wektor obserwacji (49D) części sensorycznej opiera się wyłącznie na jednostce IMU (MPU-6500) oraz przyciskach w stopach robota.
- **Uodpornienie na lukę rzeczywistości:** celowe zaszumianie toru pomiarowego i stochastyczne maskowanie sygnałów kontaktowych zamiast kosztownej randomizacji dynamiki.
- **Inicjalizacja przez klonowanie behawioralne (BC):** wstępna optymalizacja wag na analitycznych wzorcach kinematyki odwrotnej (IK).

---

## 📂 Struktura repozytorium

| Plik / Katalog | Opis |
| :--- | :--- |
| `model/` | Kompletny model kinematyczno-dynamiczny robota (pliki URDF oraz siatki kolizyjne/wizualne `.stl`). |
| `hexapod.zip` | Wytrenowany model polityki lokomocji (wagi sieci PPO w formacie Stable-Baselines3). |
| `siec.onnx` | Graf obliczeniowy aktora wyeksportowany do formatu ONNX na platformę Raspberry Pi Zero 2 W. |
| `hexapod_model_evaluation.py` | **Główny skrypt demonstracyjny** — interaktywna symulacja chodu w PyBullet z kontrolą prędkości za pomocą suwaków GUI. |
| `hexapod_learning.py` | Definicja środowiska Gym (`HexapodRLTrainingEnv`) oraz wielowątkowy proces uczenia PPO. |
| `hexapod_expert_policy.py` | Analityczny generator trajektorii chodu (kinematyka odwrotna IK) do akwizycji danych eksperckich. |
| `hexapod_behavioral_cloning.py` | Wstępne nadzorowane uczenie naśladowcze (Behavioral Cloning) na bazie danych z kinematyki analitycznej. |
| `hexapod_export_model.py` | Skrypt ekstrakcji sieci aktora z modelu PPO do deterministycznego formatu `siec.onnx`. |
| `hexapod_implementation.py` | Asynchroniczne oprogramowanie wykonawcze (`asyncio` + `onnxruntime`) dla fizycznego robota na Raspberry Pi. |

---

## 🚀 Szybki start (uruchomienie demonstracji w PyBullet)

Do uruchomienia i weryfikacji wytrenowanego modelu robota w symulatorze wystarczy wykonać poniższe kroki.

### 1. Klonowanie repozytorium
```bash
git clone https://github.com/01169704PW/Sim2Real.git
cd Sim2Real
```

### 2. Utworzenie środowiska wirtualnego i instalacja zależności
Zalecana wersja Pythona: **Python 3.10** lub **3.11**.

```bash
# Utworzenie środowiska wirtualnego
python -m venv venv

# Aktywacja środowiska:
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalacja bibliotek
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Uruchomienie demonstracji
```bash
python hexapod_model_evaluation.py
```

Po uruchomieniu otworzy się okno graficzne silnika fizycznego **PyBullet**. W prawym panelu bocznym znajdują się suwaki debugera:
- `Vy (Przod)` — sterowanie prędkością wzdłużną w zakresie `[-0.3, 0.3]` m/s,
- `Vx (Bok)` — sterowanie prędkością poprzeczną w zakresie `[-0.3, 0.3]` m/s.

Gdy suwaki są ustawione w pozycji `0.0`, robot utrzymuje stabilną postawę spoczynkową. Zmiana wartości generuje skoordynowany chód holonomiczny w zadanym kierunku.

---

## 🛠️ Odtworzenie pełnego potoku treningowego

1. **Generowanie danych eksperckich:**
   ```bash
   python hexapod_expert_policy.py
   ```

2. **Klonowanie behawioralne (inicjalizacja sieci):**
   ```bash
   python hexapod_behavioral_cloning.py
   ```

3. **Główny trening PPO ze stopniowaniem kar (Curriculum Learning):**
   ```bash
   python hexapod_learning.py
   ```

4. **Eksport wyuczonej sieci do formatu ONNX:**
   ```bash
   python hexapod_export_model.py
   ```

---

## 📹 Weryfikacja na obiekcie rzeczywistym

Działanie wyuczonego modelu po transferze bezpośrednim (*zero-shot Sim2Real*) na fizycznej platformie z procesorem Raspberry Pi Zero 2 W:

https://github.com/user-attachments/assets/a960166f-cd44-4e20-b669-4df8e238105a

