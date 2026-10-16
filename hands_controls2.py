
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pyautogui


# ============================================================
# 1. CONFIGURACOES GERAIS
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

GESTURE_THRESHOLD = 0.50
STABLE_FRAMES = 2
TOGGLE_STABLE_FRAMES = 8
TOGGLE_RELEASE_FRAMES = 5

ACTION_COOLDOWN = 0.65
SMOOTHING = 0.35

# Retangulo central da webcam.
ROI_MAX_WIDTH = 0.90
ROI_MAX_HEIGHT = 0.75

# Sensibilidade da rolagem.
SCROLL_STEP = 0.035
MAX_SCROLL_STEPS_PER_UPDATE = 3

# Aparencia.
FONT_SIZE = 0.60
FONT_THICKNESS = 2

COLOR_GREEN = (80, 240, 80)
COLOR_RED = (80, 80, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)

# Seguranca do PyAutoGUI.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

NO_GESTURE = "No gesture"


# ============================================================
# 2. AREA DE CALIBRACAO DA WEBCAM
# ============================================================

def get_control_roi(frame_width, frame_height):
    """
    Cria um retangulo central com a mesma proporcao
    da tela do monitor.

    Todo o retangulo representa a tela inteira.
    """

    screen_aspect = SCREEN_WIDTH / SCREEN_HEIGHT

    roi_width = int(frame_width * ROI_MAX_WIDTH)
    roi_height = int(roi_width / screen_aspect)

    max_height = int(frame_height * ROI_MAX_HEIGHT)

    # Ajustar o retangulo caso ele fique alto demais.
    if roi_height > max_height:
        roi_height = max_height
        roi_width = int(roi_height * screen_aspect)

    # Garantir que o retangulo caiba na imagem.
    roi_width = min(roi_width, frame_width - 1)
    roi_height = min(roi_height, frame_height - 1)

    # Centralizar.
    x1 = (frame_width - roi_width) // 2
    y1 = (frame_height - roi_height) // 2

    x2 = x1 + roi_width - 1
    y2 = y1 + roi_height - 1

    return x1, y1, x2, y2


def map_finger_to_screen(index_tip, frame_width, frame_height, roi):
    """
    Converte a ponta do indicador, em coordenadas normalizadas
    da imagem, para coordenadas da tela.

    Retorna None se o indicador estiver fora do retangulo.
    """

    x1, y1, x2, y2 = roi

    # Converter coordenadas normalizadas em pixels da webcam.
    px = index_tip[0] * (frame_width - 1)
    py = index_tip[1] * (frame_height - 1)

    # Nao permitir movimento fora da area de controle.
    inside = x1 <= px <= x2 and y1 <= py <= y2

    if not inside:
        return None

    # Normalizar a posicao dentro do retangulo.
    nx = (px - x1) / max(1, x2 - x1)
    ny = (py - y1) / max(1, y2 - y1)

    # Mapear a area inteira para a tela inteira.
    screen_x = round(nx * (SCREEN_WIDTH - 1))
    screen_y = round(ny * (SCREEN_HEIGHT - 1))

    return screen_x, screen_y


# ============================================================
# 3. RECONHECIMENTO DOS GESTOS
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

    if name in ("", "None") or score < GESTURE_THRESHOLD:
        return NO_GESTURE, score

    return name, score


# ============================================================
# 4. DESENHAR LANDMARKS E RETANGULO DE CONTROLE
# ============================================================

def draw_landmarks_on_image(bgr_image, result):
    annotated_image = np.copy(bgr_image)

    height, width, _ = annotated_image.shape

    roi = get_control_roi(width, height)
    x1, y1, x2, y2 = roi

    # Desenhar a area que corresponde ao monitor.
    cv2.rectangle(
        annotated_image,
        (x1, y1),
        (x2, y2),
        COLOR_YELLOW,
        2
    )

    cv2.putText(
        annotated_image,
        "SCREEN CONTROL AREA",
        (x1 + 8, min(y1 + 23, y2 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        COLOR_YELLOW,
        1,
        cv2.LINE_AA
    )

    for idx, hand_landmarks in enumerate(result.hand_landmarks):

        # Identificar a lateralidade.
        if idx < len(result.handedness) and result.handedness[idx]:
            hand_label = result.handedness[idx][0].category_name
        else:
            hand_label = "Unknown"

        # Identificar o gesto.
        gesture_label, score = get_gesture_info(result, idx)

        # Desenhar os pontos e as conexoes da mao.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Posicionar o texto acima da mao.
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        text_x = max(0, int(min(xs) * width))
        text_y = max(25, int(min(ys) * height) - 10)

        label = f"{hand_label} - {gesture_label}"

        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            COLOR_GREEN,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

        # Destacar a ponta do indicador (landmark 8).
        index_tip = hand_landmarks[8]

        tip_x = int(index_tip.x * (width - 1))
        tip_y = int(index_tip.y * (height - 1))

        inside_roi = (
            x1 <= tip_x <= x2
            and y1 <= tip_y <= y2
        )

        tip_color = COLOR_GREEN if inside_roi else COLOR_ORANGE

        cv2.circle(
            annotated_image,
            (tip_x, tip_y),
            8,
            tip_color,
            2,
            cv2.LINE_AA
        )

        cv2.circle(
            annotated_image,
            (tip_x, tip_y),
            3,
            tip_color,
            -1,
            cv2.LINE_AA
        )

    return annotated_image


# ============================================================
# 5. ESTADO COMPARTILHADO DO RECONHECIMENTO
# ============================================================

latest_annotated_frame = None
latest_hands = []
result_version = 0

result_lock = threading.Lock()


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

        # Landmark 8 corresponde a ponta do indicador.
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

    # Publicar o resultado para a thread principal.
    with result_lock:
        latest_annotated_frame = annotated_image
        latest_hands = hands_data
        result_version += 1


# ============================================================
# 6. CONTROLADOR DOS GESTOS
# ============================================================

class GestureMouseController:

    def __init__(self):
        self.enabled = False

        # Estabilizacao dos gestos de cada mao.
        self.candidate = {
            "Left": NO_GESTURE,
            "Right": NO_GESTURE
        }

        self.candidate_count = {
            "Left": 0,
            "Right": 0
        }

        self.stable = {
            "Left": NO_GESTURE,
            "Right": NO_GESTURE
        }

        # Controle para ligar/desligar.
        self.toggle_count = 0
        self.toggle_latched = False
        self.toggle_release_count = 0

        # Controle de repeticao das acoes.
        self.previous_combo = None
        self.last_action_time = {}

        # Arraste.
        self.drag_held = False

        # Alt_tab.
        self.alttab_held = False

        # Suavizacao do cursor.
        self.smooth_x = None
        self.smooth_y = None

        # Rolagem.
        self.scroll_last_y = None
        self.scroll_accumulator = 0.0

        self.status_detail = (
            "Mostre a palma esquerda sozinha para ativar"
        )

    # --------------------------------------------------------
    # Atualizar os gestos estaveis de cada mao.
    # --------------------------------------------------------

    def update_stability(self, hands):

        present = {
            hand["hand"]: hand
            for hand in hands
            if hand["hand"] in ("Left", "Right")
        }

        for side in ("Left", "Right"):

            hand = present.get(side)

            if hand is None:
                current = NO_GESTURE
            else:
                current = hand["gesture"]

                if hand["score"] < GESTURE_THRESHOLD:
                    current = NO_GESTURE

            if current == self.candidate[side]:
                self.candidate_count[side] += 1
            else:
                self.candidate[side] = current
                self.candidate_count[side] = 1

            if self.candidate_count[side] >= STABLE_FRAMES:
                self.stable[side] = current

        return present

    # --------------------------------------------------------
    # Alternar ativacao usando apenas a mao esquerda aberta.
    # --------------------------------------------------------

    def update_toggle(self, present):

        left_open = (
            "Left" in present
            and self.stable["Left"] == "Open_Palm"
        )

        only_left_open = (
            left_open
            and set(present.keys()) == {"Left"}
        )

        if only_left_open:
            self.toggle_count += 1
        else:
            self.toggle_count = 0

        # Para rearmar, a palma deve deixar de ser detectada
        # durante alguns resultados consecutivos.
        if not left_open and self.toggle_latched:
            self.toggle_release_count += 1

            if self.toggle_release_count >= TOGGLE_RELEASE_FRAMES:
                self.toggle_latched = False
                self.toggle_release_count = 0
        else:
            self.toggle_release_count = 0

        if (
            only_left_open
            and self.toggle_count >= TOGGLE_STABLE_FRAMES
            and not self.toggle_latched
        ):
            self.enabled = not self.enabled
            self.toggle_latched = True

            # Reiniciar o estado do cursor e da rolagem.
            self.smooth_x = None
            self.smooth_y = None
            self.reset_scroll()
            self.previous_combo = None

            if not self.enabled:
                self.release_drag()
                self.release_alttab()

            print(
                "CONTROLE ATIVADO"
                if self.enabled
                else "CONTROLE DESATIVADO"
            )

            return True

        return False

    # --------------------------------------------------------
    # Determinar a combinacao de duas maos.
    # --------------------------------------------------------

    def get_combo(self, present):

        # Todas as combinacoes abaixo exigem duas maos.
        if "Left" not in present or "Right" not in present:
            return None

        left = self.stable["Left"]
        right = self.stable["Right"]

        if right == "Closed_Fist" and left == "Pointing_Up":
            return "left_click"

        if right == "Thumb_Down" and left == "Pointing_Up":
            return "right_click"

        if right == "Thumb_Up" and left == "Pointing_Up":
            return "drag"

        if right == "Victory" and left == "Pointing_Up":
            return "scroll"

        if right == "ILoveYou" and left == "Pointing_Up":
            return "double_click"

        if right == "Closed_Fist" and left == "Victory":
            return "alt_tab"

        if right == "Thumb_Up" and left == "Victory":
            return "alt_tab_release"

        return None

    # --------------------------------------------------------
    # Mover o cursor para a posicao do indicador esquerdo.
    # --------------------------------------------------------

    def move_cursor(self, left_hand, frame_width, frame_height, roi):

        point = map_finger_to_screen(
            left_hand["index_tip"],
            frame_width,
            frame_height,
            roi
        )

        if point is None:
            self.smooth_x = None
            self.smooth_y = None

            self.status_detail = (
                "Mantenha a ponta do indicador dentro do retangulo"
            )
            return False

        target_x, target_y = point

        # Suavizar pequenas oscilacoes da mao.
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

        return True

    # --------------------------------------------------------
    # Rolagem vertical baseada no indicador.
    # --------------------------------------------------------

    def update_scroll(
        self,
        left_hand,
        frame_width,
        frame_height,
        roi
    ):

        point = map_finger_to_screen(
            left_hand["index_tip"],
            frame_width,
            frame_height,
            roi
        )

        # Fora da area, aguardar a mao voltar.
        if point is None:
            self.reset_scroll()
            self.status_detail = (
                "Rolagem: volte para dentro do retangulo"
            )
            return

        x1, y1, x2, y2 = roi

        _, normalized_y = left_hand["index_tip"]

        # Converter para a posicao vertical relativa ao ROI.
        pixel_y = normalized_y * (frame_height - 1)

        relative_y = np.clip(
            (pixel_y - y1) / max(1, y2 - y1),
            0.0,
            1.0
        )

        if self.scroll_last_y is None:
            self.scroll_last_y = relative_y
            self.scroll_accumulator = 0.0
            return

        # Se o dedo sobe, o deslocamento fica positivo.
        delta = self.scroll_last_y - relative_y

        self.scroll_accumulator += delta
        self.scroll_last_y = relative_y

        steps = int(
            self.scroll_accumulator / SCROLL_STEP
        )

        # Limitar a quantidade de rolagem em cada atualizacao.
        steps = max(
            -MAX_SCROLL_STEPS_PER_UPDATE,
            min(MAX_SCROLL_STEPS_PER_UPDATE, steps)
        )

        if steps != 0:
            pyautogui.scroll(steps)

            self.scroll_accumulator -= steps * SCROLL_STEP

        self.status_detail = (
            "Rolagem: mova o indicador para cima ou para baixo"
        )

    def reset_scroll(self):
        self.scroll_last_y = None
        self.scroll_accumulator = 0.0

    # --------------------------------------------------------
    # Iniciar e terminar o arraste.
    # --------------------------------------------------------

    def start_drag(self):

        if not self.drag_held:
            pyautogui.mouseDown(button="left")
            self.drag_held = True
            print("Arraste iniciado")

    def release_drag(self):

        if self.drag_held:
            try:
                pyautogui.mouseUp(button="left")
            finally:
                self.drag_held = False

            print("Arraste finalizado")

    def start_alttab(self):

        if not self.alttab_held:
            pyautogui.keyDown("alt")
            pyautogui.press("tab")
            self.alttab_held = True
            print("Alt + Tab iniciado")

    def release_alttab(self):

        if self.alttab_held:
            try:
                pyautogui.keyUp("alt")
            finally:
                self.alttab_held = False

            print("Alt + Tab finalizado")

    # --------------------------------------------------------
    # Cliques executados somente na entrada de uma combinacao.
    # --------------------------------------------------------

    def execute_click_action(self, combo):

        if combo not in (
            "left_click",
            "right_click",
            "double_click",
            "alt_tab",
            "alt_tab_release"
        ):
            return

        now = time.monotonic()

        previous_time = self.last_action_time.get(
            combo,
            0.0
        )

        if now - previous_time < ACTION_COOLDOWN:
            return

        if combo == "left_click":
            pyautogui.click(button="left")
            print("Clique esquerdo")

        elif combo == "right_click":
            pyautogui.click(button="right")
            print("Clique direito")

        elif combo == "double_click":
            pyautogui.doubleClick(button="left")
            print("Clique duplo")

        elif combo == "alt_tab":
            self.alttab_held = True
            pyautogui.keyDown("alt")
            pyautogui.press("tab")
            print("Alt + Tab")

        self.last_action_time[combo] = now

    # --------------------------------------------------------
    # Processar cada novo resultado do MediaPipe.
    # --------------------------------------------------------

    def process(self, hands, frame_width, frame_height, roi):

        present = self.update_stability(hands)

        # O comando de ativacao e avaliado primeiro.
        toggled = self.update_toggle(present)

        if toggled:
            self.status_detail = (
                "Controle ativado"
                if self.enabled
                else "Controle desativado"
            )
            return

        # Nenhuma acao de mouse enquanto estiver desligado.
        if not self.enabled:
            self.release_drag()
            self.release_alttab()
            self.previous_combo = None
            self.reset_scroll()
            self.smooth_x = None
            self.smooth_y = None

            self.status_detail = (
                "Mostre a palma esquerda sozinha para ativar"
            )
            return

        left_gesture = self.stable["Left"]
        right_gesture = self.stable["Right"]

        left_hand = present.get("Left")

        combo = self.get_combo(present)

        # ----------------------------------------------------
        # A. Movimento ou rolagem.
        # ----------------------------------------------------

        if combo == "scroll" and left_hand is not None:
            # Nesse modo, o movimento vertical do dedo rola
            # a pagina em vez de movimentar o cursor.
            self.update_scroll(
                left_hand,
                frame_width,
                frame_height,
                roi
            )

        else:
            self.reset_scroll()

            if (
                left_hand is not None
                and left_gesture == "Pointing_Up"
            ):
                self.move_cursor(
                    left_hand,
                    frame_width,
                    frame_height,
                    roi
                )
            else:
                self.smooth_x = None
                self.smooth_y = None

        # ----------------------------------------------------
        # B. Arrastar e soltar.
        # ----------------------------------------------------

        if combo == "drag":
            self.start_drag()
        elif combo == "alt_tab":
            self.start_alttab()
        else:
            # Soltar sempre que a combinacao terminar,
            # alguma das maos desaparecer ou mudar de gesto.
            self.release_drag()

        if combo == "alt_tab_release":
            self.release_alttab()

        # ----------------------------------------------------
        # C. Cliques por transicao de combinacao.
        # ----------------------------------------------------

        if combo != self.previous_combo:
            self.execute_click_action(combo)

        self.previous_combo = combo

        # ----------------------------------------------------
        # D. Informacoes da interface.
        # ----------------------------------------------------

        if combo == "drag":
            self.status_detail = (
                "ARRASTANDO: mantenha os dois gestos"
            )

        elif combo == "scroll":
            # update_scroll ja define a mensagem.
            pass

        elif combo == "left_click":
            self.status_detail = "Combinacao: clique esquerdo"

        elif combo == "right_click":
            self.status_detail = "Combinacao: clique direito"

        elif combo == "double_click":
            self.status_detail = "Combinacao: clique duplo"

        elif combo == "alt_tab":
            self.status_detail = "Combinacao: alt + tab"

        elif left_hand is not None and left_gesture == "Pointing_Up":
            if combo is None:
                self.status_detail = "Movendo o cursor"

        else:
            self.status_detail = (
                "Mao esquerda: mantenha o indicador levantado"
            )


# ============================================================
# 7. CONFIGURAR O GESTURE RECOGNIZER
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
# 8. CAPTURA, CONTROLE E JANELA
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
                print("Falha ao capturar a imagem.")
                break

            # Espelhar a webcam para uma visualizacao natural.
            frame = cv2.flip(frame, 1)

            frame_height, frame_width = frame.shape[:2]

            # A area de controle usa a proporcao do monitor.
            roi = get_control_roi(frame_width, frame_height)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp crescente em milissegundos.
            timestamp_ms = time.monotonic_ns() // 1_000_000

            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1

            last_timestamp_ms = timestamp_ms

            # Enviar imagem para o reconhecimento assincrono.
            recognizer.recognize_async(
                mp_image,
                timestamp_ms
            )

            # Copiar o resultado mais recente.
            with result_lock:
                if latest_annotated_frame is not None:
                    display_frame = latest_annotated_frame.copy()
                else:
                    display_frame = frame.copy()

                hands_snapshot = list(latest_hands)
                current_version = result_version

            # Processar cada resultado apenas uma vez.
            if current_version != last_processed_version:
                controller.process(
                    hands_snapshot,
                    frame_width,
                    frame_height,
                    roi
                )

                last_processed_version = current_version

            # ------------------------------------------------
            # Faixa de status da interface.
            # ------------------------------------------------

            height, width, _ = display_frame.shape

            banner_height = 72

            overlay = display_frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, banner_height),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.72,
                display_frame,
                0.28,
                0,
                display_frame
            )

            if controller.enabled:
                status = "MOUSE CONTROL: ON"
                status_color = COLOR_GREEN
            else:
                status = "MOUSE CONTROL: OFF"
                status_color = COLOR_RED

            cv2.putText(
                display_frame,
                status,
                (10, 23),
                cv2.FONT_HERSHEY_DUPLEX,
                0.62,
                status_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                controller.status_detail,
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                COLOR_WHITE,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                display_frame,
                "ESC: sair | Indicador direito: cursor",
                (10, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                COLOR_WHITE,
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
        print("Parada de seguranca do PyAutoGUI ativada.")

    finally:
        # Nunca deixar o botao pressionado ao sair.
        try:
            controller.release_drag()
        finally:
            cap.release()
            cv2.destroyAllWindows()
