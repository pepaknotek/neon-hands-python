import mediapipe as mp
import cv2
import time

# STEP 1: Importy a nastavení
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_neon_visual(img, pt1, pt2):
    """Vykreslí tyrkysový neonový efekt mezi dlaněmi."""
    color = (255, 255, 0) # BGR Tyrkysová (Cyan)
    cv2.line(img, pt1, pt2, color, 8)
    cv2.line(img, pt1, pt2, (255, 255, 255), 2)
    for pt in [pt1, pt2]:
        cv2.circle(img, pt, 20, color, 2)
        cv2.circle(img, pt, 6, (255, 255, 255), -1)

# Konfigurace detektoru
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, # Důležité pro plynulé video
    num_hands=2
)

cap = cv2.VideoCapture(0)

# Použití 'with' zajistí správné ukončení detektoru
with HandLandmarker.create_from_options(options) as landmarker:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Camera opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        # Převod z BGR (OpenCV) na RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Nové API vyžaduje mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Výpočet timestampu v milisekundách (nutné pro VIDEO mode)
        timestamp = int(time.time() * 1000)
        
        # DETEKCE
        results = landmarker.detect_for_video(mp_image, timestamp)

        # Listy pro uložení bodů obou rukou
        # hand_points[0] bude první ruka, hand_points[1] druhá ruka
        hands_data = [] 

        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                h, w, _ = frame.shape
                
                # Definujeme konečky prstů: 4=palec, 8=ukazováček, 12=prostředníček, 16=prsteníček, 20=malíček
                finger_indices = [4, 8, 12, 16, 20]
                current_hand_tips = {}

                for idx in finger_indices:
                    point = hand_lms[idx]
                    cx, cy = int(point.x * w), int(point.y * h)
                    current_hand_tips[idx] = (cx, cy)
                
                hands_data.append(current_hand_tips)

        # Pokud jsou vidět alespoň dvě ruce, propojíme odpovídající prsty
        if len(hands_data) >= 2:
            hand1 = hands_data[0]
            hand2 = hands_data[1]

            for idx in [4, 8, 12, 16, 20]:
                # Propojíme palec s palcem, ukazováček s ukazováčkem atd.
                draw_neon_visual(frame, hand1[idx], hand2[idx])

        # Neonový efekt
        if len(hands_data) >= 2:
            draw_neon_visual(frame, hands_data[0][4], hands_data[1][4]) # Propojíme jednotlivé prsty

        cv2.imshow("Neon Hands - New Tasks API", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()