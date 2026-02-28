"""
Sketched — Handwriting + ASL Finger Spelling to Text
======================================================
Two ways to input text:
  1. DRAW letters/words with your index finger (Apple Vision reads them)
  2. SIGN ASL finger spelling — hold a letter sign and it's recognized

INSTALL (one time):
  pip install pyobjc-framework-Vision pyobjc-core

GESTURES:
  ☝️  1 finger moving   → DRAW (write in the air)
  ✌️  2 fingers          → ERASE current stroke
  ✊  Fist / lower hand  → CONFIRM drawn stroke (~0.5s)
  🤟  Hold ASL sign      → letter recognized after ~1s hold

KEYS:
  SPACE  → add space
  ENTER  → new paragraph
  Z      → undo last word/letter
  C      → clear all
  S      → switch mode (Draw / ASL)
  Q      → quit

OUTPUT: sketched_output.txt (auto-saved, open in any text editor)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import threading
import os
import time
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# Apple Vision OCR
# ─────────────────────────────────────────────────────────────────────────────

def check_pyobjc():
    try:
        import Vision
        return True
    except ImportError:
        return False

HAS_VISION = check_pyobjc()

def apple_vision_recognize(image_np):
    if not HAS_VISION:
        return None
    try:
        import Vision
        from Foundation import NSURL
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, image_np)

        result_holder = [None]
        done = threading.Event()

        def handler(request, error):
            if error:
                done.set(); return
            texts = []
            for obs in (request.results() or []):
                cands = obs.topCandidates_(1)
                if cands and len(cands) > 0:
                    texts.append(str(cands[0].string()))
            result_holder[0] = " ".join(texts).strip() or None
            done.set()

        req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        req.setRecognitionLevel_(1)
        req.setUsesLanguageCorrection_(False)
        req.setMinimumTextHeight_(0.01)

        h = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
            NSURL.fileURLWithPath_(tmp.name), None)
        h.performRequests_error_([req], None)
        done.wait(timeout=6.0)
        os.unlink(tmp.name)
        return result_holder[0]
    except Exception as e:
        print(f"  Vision error: {e}")
        return None

def stroke_to_vision_image(pts, size=512, pad=50):
    if len(pts) < 2:
        return None
    arr = np.array(pts, dtype=np.float32)
    mn  = arr.min(0); mx = arr.max(0)
    rng = mx - mn; rng[rng==0] = 1
    scale = (size-2*pad) / max(rng[0], rng[1])
    arr   = (arr-mn)*scale + pad
    arr[:,0] += size/2 - (arr[:,0].min()+arr[:,0].max())/2
    arr[:,1] += size/2 - (arr[:,1].min()+arr[:,1].max())/2
    img  = np.ones((size,size,3), np.uint8)*255
    ipts = arr.astype(np.int32)
    for i in range(1, len(ipts)):
        cv2.line(img, tuple(ipts[i-1]), tuple(ipts[i]), (0,0,0), 14, cv2.LINE_AA)
    return img

def recognize_handwriting(pts):
    img = stroke_to_vision_image(pts)
    if img is not None and HAS_VISION:
        text = apple_vision_recognize(img)
        if text and text.strip():
            clean = ''.join(c for c in text.strip() if c.isprintable())
            if any(c.isalnum() or c in " '-" for c in clean):
                return clean
    return None

# ─────────────────────────────────────────────────────────────────────────────
# ASL Finger Spelling Recognition
# Uses hand landmark geometry to classify static ASL letters A-Z
# ─────────────────────────────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def finger_extended(lm, tip, pip, mcp):
    """True if finger is extended (tip above pip and mcp)."""
    return lm[tip].y < lm[pip].y and lm[pip].y < lm[mcp].y

def finger_curled(lm, tip, mcp):
    """True if fingertip is close to palm (curled)."""
    return lm[tip].y > lm[mcp].y

def thumb_extended(lm):
    """Thumb is extended if tip is far from index MCP."""
    return dist(lm[4], lm[5]) > dist(lm[3], lm[5])

def classify_asl(lm):
    """
    Classify ASL static finger spelling A-Z from hand landmarks.
    Returns letter string or None.
    
    Uses landmark distances and finger states to match standard ASL signs.
    """
    # Finger states
    idx  = finger_extended(lm, 8,  6,  5)   # index
    mid  = finger_extended(lm, 12, 10, 9)   # middle
    ring = finger_extended(lm, 16, 14, 13)  # ring
    pink = finger_extended(lm, 20, 18, 17)  # pinky
    thm  = thumb_extended(lm)               # thumb

    # Curled states (tip below MCP = fully curled)
    idx_curl  = lm[8].y  > lm[5].y
    mid_curl  = lm[12].y > lm[9].y
    ring_curl = lm[16].y > lm[13].y
    pink_curl = lm[20].y > lm[17].y

    # Fingertip spread (horizontal distance between index and pinky tips)
    spread = abs(lm[8].x - lm[20].x)

    # Key distances
    thumb_index_tip  = dist(lm[4], lm[8])
    thumb_middle_tip = dist(lm[4], lm[12])
    thumb_ring_tip   = dist(lm[4], lm[16])
    thumb_pinky_tip  = dist(lm[4], lm[20])
    index_middle_tip = dist(lm[8], lm[12])

    # Wrist reference for normalization
    wrist_mid = dist(lm[0], lm[9])
    if wrist_mid < 0.001: wrist_mid = 0.001

    # Normalize distances
    ti  = thumb_index_tip  / wrist_mid
    tm  = thumb_middle_tip / wrist_mid
    tr  = thumb_ring_tip   / wrist_mid
    tp  = thumb_pinky_tip  / wrist_mid
    im  = index_middle_tip / wrist_mid

    # ── ASL Classification ────────────────────────────────────────────────────

    # A: fist with thumb to side
    if idx_curl and mid_curl and ring_curl and pink_curl and not thm:
        return "A"

    # B: four fingers extended up, thumb folded
    if idx and mid and ring and pink and not thm:
        return "B"

    # C: curved hand, fingers form a C
    if not idx_curl and not mid_curl and not thm:
        curl_dist = dist(lm[8], lm[4])
        if 0.15 < curl_dist/wrist_mid < 0.55 and not idx and not mid:
            return "C"

    # D: index up, middle/ring/pinky curl, thumb touches middle
    if idx and not mid and not ring and not pink and tm < 0.35:
        return "D"

    # E: all fingers curled down, thumb tucked
    if idx_curl and mid_curl and ring_curl and pink_curl and thm:
        if lm[4].y > lm[8].y:
            return "E"

    # F: index and thumb touch, others extended
    if ti < 0.15 and mid and ring and pink and not idx:
        return "F"

    # G: index points sideways, thumb parallel
    if idx and not mid and not ring and not pink:
        horiz = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
        if horiz:
            return "G"

    # H: index and middle extended sideways together
    if idx and mid and not ring and not pink:
        horiz_i = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
        horiz_m = abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y)
        if horiz_i and horiz_m:
            return "H"

    # I: only pinky extended
    if not idx and not mid and not ring and pink and not thm:
        return "I"

    # J: like I but with a J motion — detected as I (motion-based J excluded)

    # K: index and middle up, thumb between them
    if idx and mid and not ring and not pink:
        if ti < 0.45 and im > 0.1:
            return "K"

    # L: index up, thumb out (L shape)
    if idx and not mid and not ring and not pink and thm:
        if lm[4].y < lm[9].y:  # thumb points outward (not up)
            return "L"

    # M: three fingers over thumb
    if idx_curl and mid_curl and ring_curl and not pink and not thm:
        return "M"

    # N: two fingers over thumb
    if idx_curl and mid_curl and not ring and not pink and not thm:
        if lm[8].y > lm[5].y and lm[12].y > lm[9].y:
            return "N"

    # O: all fingers curve to meet thumb (O shape)
    if ti < 0.18 and not idx and not mid and not ring and not pink:
        return "O"
    if ti < 0.22 and tm < 0.22 and not idx and not mid:
        return "O"

    # P: like K but pointing down
    if idx and mid and not ring and not pink and thm:
        if lm[8].y > lm[6].y:  # index pointing down
            return "P"

    # Q: like G but pointing down
    if idx and not mid and not ring and not pink and thm:
        if lm[8].y > lm[6].y:
            return "Q"

    # R: index and middle crossed
    if idx and mid and not ring and not pink:
        if lm[8].x < lm[12].x + 0.02:  # index crosses over middle
            return "R"

    # S: fist with thumb over fingers
    if idx_curl and mid_curl and ring_curl and pink_curl:
        if lm[4].x < lm[8].x and lm[4].y < lm[8].y:
            return "S"

    # T: index curled, thumb between index and middle
    if idx_curl and mid_curl and ring_curl and pink_curl:
        if lm[4].y > lm[6].y and lm[4].x > lm[5].x - 0.05:
            return "T"

    # U: index and middle up together (close)
    if idx and mid and not ring and not pink and not thm:
        if im < 0.12:
            return "U"

    # V: index and middle up, spread apart
    if idx and mid and not ring and not pink and not thm:
        if im >= 0.12:
            return "V"

    # W: index, middle, ring up
    if idx and mid and ring and not pink:
        return "W"

    # X: index hook (bent, not fully extended)
    if not idx and not mid and not ring and not pink:
        if lm[8].y < lm[6].y and lm[8].y > lm[5].y:  # partially bent
            return "X"

    # Y: thumb and pinky extended
    if not idx and not mid and not ring and pink and thm:
        return "Y"

    # Z: index traces Z — detected as index pointing (motion-based, simplified)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

detector = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.65,
)

IDX_TIP, IDX_PIP = 8,  6
MID_TIP, MID_PIP = 12, 10
RNG_TIP, RNG_PIP = 16, 14
PNK_TIP, PNK_PIP = 20, 18

def get_gesture(lm):
    idx  = lm[IDX_TIP].y < lm[IDX_PIP].y
    mid  = lm[MID_TIP].y < lm[MID_PIP].y
    rng  = lm[RNG_TIP].y < lm[RNG_PIP].y
    pnk  = lm[PNK_TIP].y < lm[PNK_PIP].y
    if idx and mid and not rng and not pnk: return "erase"
    if idx and not mid:                     return "draw"
    if not any([idx,mid,rng,pnk]):          return "fist"
    return "pause"

# ─────────────────────────────────────────────────────────────────────────────
# HUD helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_rect(img, x1, y1, x2, y2, col, alpha=0.72):
    roi = img[y1:y2, x1:x2]; bg = roi.copy()
    cv2.rectangle(bg, (0,0), (x2-x1, y2-y1), col, -1)
    cv2.addWeighted(bg, alpha, roi, 1-alpha, 0, roi)
    img[y1:y2, x1:x2] = roi

def put_txt(img, text, pos, scale=0.55, color=(255,255,255), bold=False):
    font = cv2.FONT_HERSHEY_SIMPLEX; th = 2 if bold else 1
    cv2.putText(img, text, (pos[0]+1,pos[1]+1), font, scale, (0,0,0), th+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, th, cv2.LINE_AA)

def wrap_text(text, max_chars=30):
    words = text.split()
    lines, line = [], ""
    for w in words:
        if len(line)+len(w)+1 <= max_chars:
            line += (" " if line else "") + w
        else:
            if line: lines.append(line)
            line = w
    if line: lines.append(line)
    return lines if lines else [""]

# ─────────────────────────────────────────────────────────────────────────────
# File I/O
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE = "sketched_output.txt"

def save_output(paragraphs):
    with open(OUTPUT_FILE, 'w') as f:
        for p in paragraphs:
            f.write(p.strip() + "\n\n")

def load_output():
    if not os.path.exists(OUTPUT_FILE):
        return [""]
    with open(OUTPUT_FILE) as f:
        raw = f.read()
    paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return paras if paras else [""]

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera"); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read camera"); return

    H, W = frame.shape[:2]
    sketch = np.zeros((H, W, 4), dtype=np.uint8)

    paragraphs   = load_output()
    word_history = []
    last_word    = ""

    # Draw mode state
    stroke_pts   = []
    prev_pt      = None
    was_drawing  = False
    pause_frames = 0
    CONFIRM_FRAMES = 18
    ERASE_R        = 35

    recognizing   = False
    pending_word  = [None]

    # ASL state
    asl_mode           = False   # False=draw, True=ASL
    asl_buffer         = deque(maxlen=20)   # recent ASL predictions
    asl_hold_start     = None
    asl_last_letter    = ""
    asl_last_added     = ""
    asl_cooldown       = 0       # frames to wait before next ASL letter
    ASL_HOLD_FRAMES    = 22      # frames to hold sign before accepting
    ASL_COOLDOWN_FRAMES= 35      # frames between letters

    gesture = "none"

    WIN = "Sketched  |  S=switch mode  SPACE  ENTER  Z=undo  C=clear  Q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.moveWindow(WIN, 0, 40)

    mode_str = "Apple Vision 🍎" if HAS_VISION else "Shape detection"
    print(f"\n✅  Sketched ready!")
    print(f"  Handwriting mode: {mode_str}")
    print(f"  ASL mode: built-in landmark classifier (A-Y)")
    print(f"  Output → {os.path.abspath(OUTPUT_FILE)}")
    print("  S = switch Draw/ASL  |  SPACE  ENTER  Z  C  Q\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)

        gesture   = "none"
        finger_pt = None
        lm_list   = None

        if result.multi_hand_landmarks:
            lm_list = result.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame,
                result.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style())
            tip = lm_list[IDX_TIP]
            finger_pt = (int(tip.x*W), int(tip.y*H))
            gesture   = get_gesture(lm_list)

        # ══════════════════════════════════════════════════════════════════
        # DRAW MODE
        # ══════════════════════════════════════════════════════════════════
        if not asl_mode:

            if gesture == "draw" and finger_pt and not recognizing:
                pause_frames = 0
                if prev_pt:
                    cv2.line(sketch, prev_pt, finger_pt, (80,190,255,220), 5, cv2.LINE_AA)
                stroke_pts.append(finger_pt)
                prev_pt     = finger_pt
                was_drawing = True
                cv2.circle(frame, finger_pt, 13, (0,245,196), 2, cv2.LINE_AA)
                cv2.circle(frame, finger_pt,  4, (0,245,196), -1)

            elif gesture == "erase" and finger_pt:
                prev_pt = None
                cv2.circle(sketch, finger_pt, ERASE_R, (0,0,0,0), -1)
                stroke_pts = [p for p in stroke_pts
                              if math.hypot(p[0]-finger_pt[0], p[1]-finger_pt[1]) > ERASE_R]
                cv2.circle(frame, finger_pt, ERASE_R, (120,80,255), 2, cv2.LINE_AA)
                cv2.circle(frame, finger_pt, 5, (120,80,255), -1)
                was_drawing = len(stroke_pts) > 0

            else:
                prev_pt = None
                if was_drawing and len(stroke_pts) > 8 and not recognizing:
                    pause_frames += 1
                    if pause_frames >= CONFIRM_FRAMES:
                        recognizing = True
                        snap = list(stroke_pts)

                        def run(pts):
                            word = recognize_handwriting(pts) or "?"
                            pending_word[0] = word

                        threading.Thread(target=run, args=(snap,), daemon=True).start()
                        sketch[:] = 0
                        stroke_pts.clear()
                        was_drawing  = False
                        pause_frames = 0
                elif gesture not in ("draw","erase") and not was_drawing:
                    pause_frames = 0

            if recognizing and pending_word[0] is not None:
                word = pending_word[0]
                pending_word[0] = None
                recognizing     = False
                if word and word != "?":
                    last_word = word
                    word_history.append((len(paragraphs)-1, paragraphs[-1]))
                    paragraphs[-1] = (paragraphs[-1]+" "+word).strip() if paragraphs[-1] else word
                    save_output(paragraphs)
                    print(f"  ✏️  '{word}'")

        # ══════════════════════════════════════════════════════════════════
        # ASL MODE
        # ══════════════════════════════════════════════════════════════════
        else:
            sketch[:] = 0   # no drawing in ASL mode

            if asl_cooldown > 0:
                asl_cooldown -= 1

            if lm_list:
                letter = classify_asl(lm_list)
                asl_buffer.append(letter)

                # Majority vote over buffer
                counts = {}
                for l in asl_buffer:
                    if l: counts[l] = counts.get(l,0) + 1
                best = max(counts, key=counts.get) if counts else None
                best_count = counts.get(best, 0) if best else 0
                asl_last_letter = best or ""

                # Confirm letter if held consistently
                if best and best_count >= len(asl_buffer)*0.7:
                    if asl_hold_start is None:
                        asl_hold_start = time.time()
                    hold_frames = int((time.time()-asl_hold_start)*30)

                    # Draw hold progress arc on fingertip
                    if finger_pt:
                        progress = min(hold_frames / ASL_HOLD_FRAMES, 1.0)
                        angle    = int(progress * 360)
                        cv2.ellipse(frame, finger_pt, (25,25), -90, 0, angle,
                                    (0,245,196), 3, cv2.LINE_AA)
                        cv2.circle(frame, finger_pt, 6, (0,245,196), -1)

                    if hold_frames >= ASL_HOLD_FRAMES and asl_cooldown == 0:
                        # Accept letter
                        last_word = best
                        asl_last_added = best
                        word_history.append((len(paragraphs)-1, paragraphs[-1]))
                        paragraphs[-1] += best
                        save_output(paragraphs)
                        print(f"  🤟  ASL '{best}'")
                        asl_hold_start  = None
                        asl_cooldown    = ASL_COOLDOWN_FRAMES
                        asl_buffer.clear()
                else:
                    asl_hold_start = None
            else:
                asl_buffer.clear()
                asl_hold_start  = None
                asl_last_letter = ""

        # ── Build display ───────────────────────────────────────────────────
        PANEL_W = 340
        cam_w   = W - PANEL_W

        a = sketch[:,:cam_w,3:4].astype(np.float32)/255.0
        cam = np.clip(
            sketch[:,:cam_w,:3].astype(np.float32)*a +
            frame[:,:cam_w].astype(np.float32)*(1-a),
            0,255).astype(np.uint8)

        # Top bar
        overlay_rect(cam, 0,0,cam_w,54,(10,10,20),0.82)

        # Mode badge
        mode_col  = (0,180,255) if asl_mode else (0,245,196)
        mode_label= "🤟 ASL MODE" if asl_mode else "✏️  DRAW MODE"
        put_txt(cam, mode_label, (14,34), scale=0.7, color=mode_col, bold=True)
        put_txt(cam, "S=switch", (cam_w-120,34), scale=0.45, color=(80,80,110))

        if not asl_mode:
            # Draw mode status
            if recognizing:
                dots = "."*(int(time.time()*3)%4)
                put_txt(cam, f"🍎 Reading{dots}", (250,34), scale=0.55, color=(0,200,255))
            elif gesture=="draw":
                put_txt(cam, f"☝ {len(stroke_pts)}pts", (250,34), scale=0.5, color=(0,245,196))
            elif was_drawing and pause_frames>0:
                put_txt(cam, "✊ confirming...", (250,34), scale=0.5, color=(0,200,255))

            # Confirm progress bar
            if was_drawing and pause_frames>0 and not recognizing:
                bw = int(pause_frames/CONFIRM_FRAMES*(cam_w-40))
                cv2.rectangle(cam,(20,H-48),(20+bw,H-28),(0,210,100),-1)
                cv2.rectangle(cam,(20,H-48),(cam_w-20,H-28),(50,50,50),1)
                put_txt(cam,"lower hand / fist to confirm",
                        (25,H-30),scale=0.40,color=(180,255,180))

        else:
            # ASL mode: show current detected letter BIG
            if asl_last_letter:
                # Giant letter display
                letter_scale = 3.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw,th),_ = cv2.getTextSize(asl_last_letter, font, letter_scale, 6)
                lx = cam_w//2 - tw//2
                ly = H//2 + th//2
                # Background circle
                cv2.circle(cam,(cam_w//2, H//2), 80, (20,20,30),-1)
                cv2.circle(cam,(cam_w//2, H//2), 80, mode_col, 2)
                cv2.putText(cam, asl_last_letter, (lx,ly),
                            font, letter_scale, mode_col, 6, cv2.LINE_AA)

            # Hold progress info
            if asl_hold_start and asl_last_letter:
                pct = min(int((time.time()-asl_hold_start)/( ASL_HOLD_FRAMES/30)*100),100)
                put_txt(cam,f"Hold '{asl_last_letter}'... {pct}%",
                        (14,H-48),scale=0.55,color=mode_col)

            if asl_last_added:
                put_txt(cam,f"Added: '{asl_last_added}'",
                        (cam_w-200,H-48),scale=0.52,color=(0,220,255))

        # Bottom bar
        overlay_rect(cam,0,H-26,cam_w,H,(10,10,20),0.80)
        if asl_mode:
            put_txt(cam,"Hold ASL sign to add letter  |  SPACE  ENTER  Z=undo  Q=quit",
                    (14,H-8),scale=0.38,color=(80,80,110))
        else:
            put_txt(cam,"☝draw  ✌erase  ✊confirm  |  SPACE  ENTER  Z=undo  Q=quit",
                    (14,H-8),scale=0.38,color=(80,80,110))

        if last_word:
            put_txt(cam,f"Last: {last_word}",(cam_w-200,H-8),scale=0.38,color=(0,180,180))

        # ── Text panel ─────────────────────────────────────────────────────
        panel = np.ones((H,PANEL_W,3),np.uint8)*15
        cv2.rectangle(panel,(0,0),(PANEL_W-1,H-1),(50,50,50),1)

        put_txt(panel,"OUTPUT",(12,28),scale=0.65,color=(0,245,196),bold=True)
        mode_info = "ASL" if asl_mode else ("Vision" if HAS_VISION else "Geometry")
        put_txt(panel,mode_info,(160,28),scale=0.40,color=(80,80,100))
        cv2.line(panel,(0,38),(PANEL_W,38),(50,50,50),1)

        y = 56
        for pi,para in enumerate(paragraphs):
            is_cur = (pi==len(paragraphs)-1)
            col    = (235,235,235) if is_cur else (120,120,120)
            lines  = wrap_text(para if para else "(empty)")
            for li,line in enumerate(lines):
                if y > H-65: break
                disp = line
                if is_cur and li==len(lines)-1 and int(time.time()*2)%2==0:
                    disp += "|"
                put_txt(panel,disp,(12,y),scale=0.48,color=col)
                y += 22
            y += 10
            if y > H-65: break

        # ASL reference chart (small, bottom of panel)
        if asl_mode:
            cv2.line(panel,(0,H-130),(PANEL_W,H-130),(50,50,50),1)
            put_txt(panel,"ASL Quick Ref:",(12,H-115),scale=0.38,color=(100,100,120))
            refs = ["A=fist  B=4up  C=curve","D=idx  I=pink  L=L-shape",
                    "V=peace  W=3up  Y=Y-shape","O=circle  S=thumb-fist"]
            for ri,r in enumerate(refs):
                put_txt(panel,r,(12,H-95+ri*20),scale=0.30,color=(80,80,100))

        cv2.line(panel,(0,H-30),(PANEL_W,H-30),(50,50,50),1)
        tw2 = sum(len(p) for p in paragraphs)
        put_txt(panel,f"chars: {tw2}  paras: {len(paragraphs)}",
                (12,H-15),scale=0.35,color=(70,70,70))

        combined = np.hstack([cam, panel])
        cv2.imshow(WIN, combined)

        # ── Keys ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'),27):
            break
        elif key == ord('s'):
            asl_mode = not asl_mode
            sketch[:] = 0; stroke_pts.clear()
            asl_buffer.clear(); asl_hold_start=None
            was_drawing=False; pause_frames=0
            print(f"  🔄  Switched to {'ASL' if asl_mode else 'Draw'} mode")
        elif key == ord(' '):
            word_history.append((len(paragraphs)-1, paragraphs[-1]))
            paragraphs[-1] += " "
            save_output(paragraphs)
        elif key in (13,10):
            word_history.append((len(paragraphs)-1, paragraphs[-1]))
            paragraphs.append("")
            save_output(paragraphs)
            print("  ↵  New paragraph")
        elif key == ord('z') and word_history:
            pi,old = word_history.pop()
            if pi < len(paragraphs):
                paragraphs[pi] = old
            while len(paragraphs)>1 and paragraphs[-1]=="":
                paragraphs.pop()
            save_output(paragraphs)
            print("  ↩️  Undo")
        elif key == ord('c'):
            word_history.clear(); paragraphs=[""]
            sketch[:]=0; stroke_pts.clear()
            asl_buffer.clear(); asl_hold_start=None
            last_word=""; was_drawing=False; pause_frames=0
            save_output(paragraphs)
            print("  🗑  Cleared")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print(f"\n👋  Saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()
