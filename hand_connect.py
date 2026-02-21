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
        
        # DETEKCE (místo .process používáme .detect_for_video)
        results = landmarker.detect_for_video(mp_image, timestamp)

        hand_points = []

        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                h, w, _ = frame.shape
                # Landmark 9 (střed dlaně)
                # POZOR: V novém API je to seznam objektů, ne proto-buffer
                landmark = hand_lms[9]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                hand_points.append((cx, cy))
                
                # Vykreslení bodů (jednoduché kruhy, protože drawing_utils zlobí)
                for lm in hand_lms:
                    ix, iy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)

        # Neonový efekt
        if len(hand_points) >= 2:
            draw_neon_visual(frame, hand_points[0], hand_points[1])

        cv2.imshow("Neon Hands - New Tasks API", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()