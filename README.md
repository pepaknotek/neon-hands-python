🖐️ Neon Hands Python
Tento projekt využívá pokročilé počítačové vidění k vytvoření interaktivního neonového efektu. Program detekuje konečky prstů obou rukou a v reálném čase je propojuje svítícími čarami, čímž vytváří futuristický vizuální zážitek.

Ukázka funkčního propojení prstů pomocí MediaPipe Tasks API.

<img width="485" height="379" alt="image" src="https://github.com/user-attachments/assets/09e52577-ad6e-4d1e-a31e-b52729e6f5f0" />


✨ Funkce
Multi-finger Tracking: Sleduje a propojuje všech 5 prstů (palec, ukazováček, prostředníček, prsteníček a malíček).

Neon Visuals: Simulace neonové záře pomocí vrstvení čar v OpenCV.

Optimalizováno pro 0.10.32: Využívá nejnovější MediaPipe Tasks API pro vysokou přesnost.

🛠️ Použité technologie
MediaPipe (Tasks API): Engine pro detekci klíčových bodů ruky.

OpenCV: Zpracování obrazu, obsluha kamery a vykreslování grafiky.

NumPy: Efektivní práce s maticemi obrazových dat.

Python 3.12+: Projekt je plně kompatibilní s nejnovějšími verzemi Pythonu.

🚀 Instalace a spuštění
Klonování:

Bash
git clone https://github.com/vladimír-pepernik/neon-hands-python.git
Příprava prostředí:
Ujisti se, že máš aktivní virtuální prostředí a nainstalované závislosti:

Bash
pip install -r requirements.txt
Model:
Stáhni soubor hand_landmarker.task a vlož ho do složky projektu.

Spuštění:

Bash
python main.py

📝 Poznámky k vývoji
Při vývoji byl kladen důraz na překonání problémů s kompatibilitou mediapipe.solutions v novějších verzích Pythonu pomocí přímého importu mediapipe.tasks.
