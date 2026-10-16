import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pyautogui

# ============================================================
# 1. CONFIGURACOES
# ============================================================

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MODEL_PATH = "gesture_recognizer.task"

MARGIN = 10
FONT_SIZE = 0.65
FONT_THICKNESS = 2

GESTURE_THRESHOLD = 0.65
STABLE_FRAMES = 2
TOGGLE_STABLE_FRAMES = 3
TOGGLE_RELEASE_FRAMES = 5
ACTION_COOLDOWN = 0.7
SMOOTHING = 0.35
CURSOR_MARGIN = 15

# Mantem a protecao de seguranca do PyAutoGUI.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Cor dos textos em BGR.
TEXT_COLOR = (88, 205, 54)

NO_GESTURE = "No gesture"


# ============================================================
# 2. ESTADO COMPARTILHADO DOS RESULTADOS
# ============================================================

latest_annotated_frame = None
latest_hands = []
result_version = 0

result_lock = threading.Lock()


# ============================================================
# 3. EXTRAIR O GESTO RECONHECIDO
# ============================================================

def get_gesture_info(result, idx):
    if idx >= len(result.gestures):
        return NO_GESTURE, 0.0

    categories = result.gestures[idx]

    if not categories:
        return NO_GESTURE, 0.0

    gesture = categories[0]
    name = gesture.category_name
    score = float(gesture.score)

    if name == "None" or score < GESTURE_THRESHOLD:
        return NO_GESTURE, score

    return name, score


# ============================================================
# 4. DESENHAR LANDMARKS, LATERALIDADE E GESTO
# ============================================================

def draw_landmarks_on_image(bgr_image, result):
    annotated_image = np.copy(bgr_image)

    height, width, _ = annotated_image.shape

    for idx, hand_landmarks in enumerate(result.hand_landmarks):

        # Identificacao da mao.
        handedness = result.handedness[idx]

        if handedness:
            hand_label = handedness[0].category_name
        else:
            hand_label = "Unknown"

        # Identificacao do gesto.
        gesture_label, score = get_gesture_info(result, idx)

        # Desenhar os 21 pontos e as conexoes.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Calcular a posicao do texto.
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]

        text_x = int(min(xs) * width)
        text_y = int(min(ys) * height) - MARGIN

        label = f"{hand_label} - {gesture_label}"

        (text_width, _), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            FONT_THICKNESS
        )

        text_x = max(0, min(text_x, width - text_width - 5))
        text_y = max(25, text_y)

        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image


# ============================================================
# 5. CONTROLADOR DO MOUSE POR GESTOS
# ============================================================

class GestureMouseController:

    def __init__(self):
        # O controle comeca desligado.
        self.enabled = False

        # Controle de estabilidade dos gestos por mao.
        self.candidate_gesture = {
            "Left": NO_GESTURE,
            "Right": NO_GESTURE
        }

        self.candidate_count = {
            "Left": 0,
            "Right": 0
        }

        self.stable_gesture = {
            "Left": NO_GESTURE,
            "Right": NO_GESTURE
        }

        # Controle para alternar a ativacao com palma aberta.
        self.open_palm_count = 0
        self.open_palm_latched = False
        self.open_palm_release_count = 0

        # Impede repeticao rapida de acoes.
        self.last_action_time = {}

        # Suavizacao do movimento do cursor.
        self.smooth_x = None
        self.smooth_y = None

    def process(self, hands):

        was_enabled = self.enabled

        # ----------------------------------------------------
        # A. ATIVAR OU DESATIVAR COM OPEN_PALM
        # ----------------------------------------------------

        open_palm_detected = any(
            hand["gesture"] == "Open_Palm"
            and hand["score"] >= GESTURE_THRESHOLD
            for hand in hands
        )

        if open_palm_detected:
            self.open_palm_count += 1
            self.open_palm_release_count = 0

            # Exige varios resultados consecutivos para alternar.
            if (
                self.open_palm_count >= TOGGLE_STABLE_FRAMES
                and not self.open_palm_latched
            ):
                self.enabled = not self.enabled
                self.open_palm_latched = True

                self.smooth_x = None
                self.smooth_y = None

                if self.enabled:
                    print("CONTROLE DO MOUSE ATIVADO")
                else:
                    print("CONTROLE DO MOUSE DESATIVADO")

        else:
            self.open_palm_count = 0

            # Aguarda a mao deixar de apresentar palma aberta
            # antes de permitir uma nova alternancia.
            if self.open_palm_latched:
                self.open_palm_release_count += 1

                if (
                    self.open_palm_release_count
                    >= TOGGLE_RELEASE_FRAMES
                ):
                    self.open_palm_latched = False
                    self.open_palm_release_count = 0

        # ----------------------------------------------------
        # B. IDENTIFICAR MUDANCAS DE GESTO
        # ----------------------------------------------------

        present_hands = set()
        entered_gestures = []

        for hand in hands:
            side = hand["hand"]

            if side not in ("Left", "Right"):
                continue

            present_hands.add(side)

            if hand["score"] >= GESTURE_THRESHOLD:
                current = hand["gesture"]
            else:
                current = NO_GESTURE

            if current == self.candidate_gesture[side]:
                self.candidate_count[side] += 1
            else:
                self.candidate_gesture[side] = current
                self.candidate_count[side] = 1

            # Atualizar o gesto estavel.
            if self.candidate_count[side] >= STABLE_FRAMES:
                previous = self.stable_gesture[side]

                if previous != current:
                    self.stable_gesture[side] = current

                    if current != NO_GESTURE:
                        entered_gestures.append((side, current))

        # Se uma mao desaparecer, resetar seu estado gradualmente.
        for side in ("Left", "Right"):
            if side in present_hands:
                continue

            if self.candidate_gesture[side] != NO_GESTURE:
                self.candidate_gesture[side] = NO_GESTURE
                self.candidate_count[side] = 1
            else:
                self.candidate_count[side] += 1

            if self.candidate_count[side] >= STABLE_FRAMES:
                self.stable_gesture[side] = NO_GESTURE

        # ----------------------------------------------------
        # C. EXECUTAR CLIQUES E ROLAGEM
        # ----------------------------------------------------

        # Evita executar uma acao durante a mesma atualizacao
        # que acabou de ativar ou desativar o sistema.
        if was_enabled and self.enabled:
            for side, gesture in entered_gestures:
                self.execute_action(gesture)

        # ----------------------------------------------------
        # D. MOVER O CURSOR COM POINTING_UP
        # ----------------------------------------------------

        if not self.enabled:
            self.smooth_x = None
            self.smooth_y = None
            return

        pointing_hand = next(
            (
                hand for hand in hands
                if hand["hand"] in ("Left", "Right")
                and hand["gesture"] == "Pointing_Up"
                and hand["score"] >= GESTURE_THRESHOLD
                and self.stable_gesture[hand["hand"]]
                    == "Pointing_Up"
            ),
            None
        )

        if pointing_hand is None:
            self.smooth_x = None
            self.smooth_y = None
            return

        # O landmark 8 e a ponta do dedo indicador.
        index_x, index_y = pointing_hand["index_tip"]

        # Limitar a area de controle para evitar os cantos
        # absolutos da tela.
        index_x = np.clip(index_x, 0.04, 0.96)
        index_y = np.clip(index_y, 0.04, 0.96)

        target_x = np.interp(
            index_x,
            [0.04, 0.96],
            [CURSOR_MARGIN, SCREEN_WIDTH - CURSOR_MARGIN]
        )

        target_y = np.interp(
            index_y,
            [0.04, 0.96],
            [CURSOR_MARGIN, SCREEN_HEIGHT - CURSOR_MARGIN]
        )

        # Suavizar o movimento para reduzir tremores.
        if self.smooth_x is None:
            self.smooth_x = target_x
            self.smooth_y = target_y
        else:
            self.smooth_x += SMOOTHING * (
                target_x - self.smooth_x
            )
            self.smooth_y += SMOOTHING * (
                target_y - self.smooth_y
            )

        pyautogui.moveTo(
            int(self.smooth_x),
            int(self.smooth_y)
        )

    def execute_action(self, gesture):

        now = time.monotonic()
        previous_time = self.last_action_time.get(gesture, 0)

        # Evita cliques repetidos causados por oscilacoes
        # na classificacao do gesto.
        if now - previous_time < ACTION_COOLDOWN:
            return

        if gesture == "Closed_Fist":
            pyautogui.click()
            print("Acao: clique esquerdo")

        elif gesture == "Victory":
            pyautogui.click(button="right")
            print("Acao: clique direito")

        elif gesture == "Thumb_Up":
            pyautogui.click(clicks=2, interval=0.15)
            print("Acao: clique duplo")

        elif gesture == "Thumb_Down":
            pyautogui.scroll(-3)
            print("Acao: rolagem para baixo")

        else:
            return

        self.last_action_time[gesture] = now


# ============================================================
# 6. CALLBACK DO GESTURE RECOGNIZER
# ============================================================

def print_result(
    result: GestureRecognizerResult,
    output_image: mp.Image,
    timestamp_ms: int
):
    global latest_annotated_frame
    global latest_hands
    global result_version

    rgb_image = output_image.numpy_view()

    bgr_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR
    )

    annotated_image = draw_landmarks_on_image(
        bgr_image,
        result
    )

    hands_data = []

    for idx, hand_landmarks in enumerate(result.hand_landmarks):

        if idx < len(result.handedness) and result.handedness[idx]:
            hand_label = result.handedness[idx][0].category_name
        else:
            hand_label = "Unknown"

        gesture_name, gesture_score = get_gesture_info(
            result,
            idx
        )

        # Guardar a ponta do dedo indicador.
        index_tip = hand_landmarks[8]

        hands_data.append({
            "hand": hand_label,
            "gesture": gesture_name,
            "score": gesture_score,
            "index_tip": (
                float(index_tip.x),
                float(index_tip.y)
            )
        })

    # Publicar o resultado mais recente para a thread principal.
    with result_lock:
        latest_annotated_frame = annotated_image
        latest_hands = hands_data
        result_version += 1


# ============================================================
# 7. CONFIGURAR O RECONHECEDOR
# ============================================================

options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
)


# ============================================================
# 8. EXECUTAR A WEBCAM E O CONTROLE
# ============================================================

controller = GestureMouseController()
last_timestamp_ms = -1
last_processed_version = 0

with GestureRecognizer.create_from_options(options) as recognizer:

    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Nao foi possivel abrir a webcam."
            )

        while True:
            success, frame = cap.read()

            if not success:
                print("Falha ao capturar a imagem da webcam.")
                break

            # Espelhar para uma visualizacao natural.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp crescente para o modo LIVE_STREAM.
            timestamp_ms = time.monotonic_ns() // 1_000_000

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            # Enviar para o reconhecimento assincrono.
            recognizer.recognize_async(
                mp_image,
                timestamp_ms
            )

            # Obter o resultado mais recente.
            with result_lock:
                if latest_annotated_frame is not None:
                    display_frame = latest_annotated_frame.copy()
                else:
                    display_frame = frame.copy()

                hands_snapshot = list(latest_hands)
                current_version = result_version

            # Processar cada resultado novo apenas uma vez.
            if current_version != last_processed_version:
                controller.process(hands_snapshot)
                last_processed_version = current_version

            # Faixa de status.
            height, width, _ = display_frame.shape

            overlay = display_frame.copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (width, 68),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.60,
                display_frame,
                0.40,
                0,
                display_frame
            )

            if controller.enabled:
                status_text = "MOUSE CONTROL: ON"
                status_color = (80, 240, 80)
            else:
                status_text = "MOUSE CONTROL: OFF"
                status_color = (80, 80, 255)

            cv2.putText(
                display_frame,
                status_text,
                (10, 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                "Open Palm: toggle | Point Up: move | Fist: click",
                (10, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.imshow(
                "Gesture Mouse Control",
                display_frame
            )

            # ESC encerra o programa.
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except pyautogui.FailSafeException:
        print(
            "Parada de seguranca ativada pelo PyAutoGUI."
        )

    finally:
        cap.release()
        cv2.destroyAllWindows()