import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import json
import time
import numpy as np
from collections import deque

class GestureRecognizer:
    def __init__(self):
        # Configuration MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        
        # Caméra
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Charger les signatures
        self.data_dir = "gesture_data"
        self.templates = {}
        self.load_templates()
        
        # État
        self.recognizing = False
        self.recognition_data = []
        self.recognition_start = 0
        self.recognition_duration = 5.0
        self.result = None
        self.result_score = 0
        
        # Historique pour lissage des résultats
        self.result_history = deque(maxlen=5)
        
        print("="*50)
        print("🤟 RECONNAISSANCE DES GESTES")
        print("="*50)
        print(f"{len(self.templates)} gestes charges")
        print("\nCOMMANDES:")
        print("   ESPACE : Commencer la reconnaissance (5 secondes)")
        print("   ESC : Quitter")
        print("="*50)
    
    def load_templates(self):
        """Charge les signatures des gestes enregistrés"""
        if not os.path.exists(self.data_dir):
            print("❌ Aucune donnee! Lancez rec.py d'abord")
            return
        
        for f in os.listdir(self.data_dir):
            if f.endswith('_sig.json'):
                filepath = os.path.join(self.data_dir, f)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    self.templates[data["gesture"]] = data["signature"]
                    print(f"✅ {data['gesture']}")
    
    def get_hand_data(self, hand_landmarks):
        """Extrait les points de la main"""
        if not hand_landmarks:
            return None
        
        data = []
        for lm in hand_landmarks.landmark:
            data.append([lm.x, lm.y, lm.z])
        return data
    
    def get_face_data(self, face_landmarks):
        """Extrait les points clés du visage"""
        if not face_landmarks:
            return None
        
        key_points = [1, 33, 61, 93, 133, 152, 157, 173, 199, 234, 263, 291, 299, 334, 362]
        data = []
        for idx in key_points:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                data.append([lm.x, lm.y, lm.z])
        return data
    
    def compute_hand_angles(self, hand_points):
        """Calcule les angles entre les doigts"""
        if not hand_points or len(hand_points) < 21:
            return []
        
        finger_bones = [
            (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20)
        ]
        
        angles = []
        for p1, p2 in finger_bones:
            if p1 < len(hand_points) and p2 < len(hand_points):
                point1 = np.array(hand_points[p1])
                point2 = np.array(hand_points[p2])
                diff = point2 - point1
                angle = np.arctan2(diff[1], diff[0])
                angles.append(angle)
        
        return angles
    
    def compute_hand_distances(self, hand_points):
        """Calcule les distances relatives entre les points de la main"""
        if not hand_points or len(hand_points) < 21:
            return []
        
        palm_base = np.array(hand_points[0])
        
        distances = []
        for i in range(1, 21):
            point = np.array(hand_points[i])
            dist = np.linalg.norm(point - palm_base)
            distances.append(dist)
        
        if distances:
            max_dist = max(distances)
            if max_dist > 0:
                distances = [d / max_dist for d in distances]
        
        return distances
    
    def compute_hand_features(self, hand_points):
        """Extrait un vecteur de caractéristiques riche pour la main"""
        if not hand_points:
            return None
        
        features = []
        
        # Angles des doigts
        angles = self.compute_hand_angles(hand_points)
        features.extend(angles)
        
        # Distances relatives
        distances = self.compute_hand_distances(hand_points)
        features.extend(distances)
        
        # Position relative des bouts de doigts
        fingertips = [4, 8, 12, 16, 20]
        palm_base = np.array(hand_points[0])
        
        for tip in fingertips:
            if tip < len(hand_points):
                tip_pos = np.array(hand_points[tip])
                relative_pos = tip_pos - palm_base
                features.extend([relative_pos[0], relative_pos[1], relative_pos[2]])
        
        # Courbure des doigts
        finger_bases = [1, 5, 9, 13, 17]
        for base, tip in zip(finger_bases, fingertips):
            if base < len(hand_points) and tip < len(hand_points):
                base_pos = np.array(hand_points[base])
                tip_pos = np.array(hand_points[tip])
                finger_len = np.linalg.norm(tip_pos - base_pos)
                features.append(finger_len)
        
        return features
    
    def compute_live_signature(self, frames_data):
        """
        Calcule la signature du geste en cours à partir des frames capturées
        """
        if not frames_data or len(frames_data) < 10:
            return None
        
        hand_features = {"Right": [], "Left": []}
        face_features = []
        
        for frame_data in frames_data:
            for hand_type in ["Right", "Left"]:
                hand_points = frame_data.get("hands", {}).get(hand_type)
                if hand_points:
                    features = self.compute_hand_features(hand_points)
                    if features:
                        hand_features[hand_type].append(features)
            
            face = frame_data.get("face")
            if face:
                face_flat = []
                for point in face:
                    face_flat.extend([point[0], point[1], point[2]])
                if face_flat:
                    face_features.append(face_flat)
        
        signature = {
            "frames_original": len(frames_data),
            "hands": {}
        }
        
        for hand_type in ["Right", "Left"]:
            if hand_features[hand_type]:
                features_array = np.array(hand_features[hand_type])
                signature["hands"][hand_type] = {
                    "mean": features_array.mean(axis=0).tolist(),
                    "std": features_array.std(axis=0).tolist(),
                    "num_frames": len(features_array),
                    "keyframes": features_array[::max(1, len(features_array)//8)].tolist()
                }
            else:
                signature["hands"][hand_type] = None
        
        if face_features:
            face_array = np.array(face_features)
            signature["face"] = {
                "mean": face_array.mean(axis=0).tolist(),
                "std": face_array.std(axis=0).tolist()
            }
        else:
            signature["face"] = None
        
        return signature
    
    def compare_signatures(self, template_sig, live_sig):
        """
        Compare deux signatures de gestes avec des métriques pondérées
        """
        if not template_sig or not live_sig:
            return 0.0
        
        scores = []
        weights = []
        
        # Comparer chaque main
        for hand_type in ["Right", "Left"]:
            template_hand = template_sig["hands"].get(hand_type)
            live_hand = live_sig["hands"].get(hand_type)
            
            if template_hand and live_hand:
                # Comparer les moyennes
                template_mean = np.array(template_hand["mean"])
                live_mean = np.array(live_hand["mean"])
                
                # Distance cosinus (meilleure que euclidienne pour les vecteurs de caractéristiques)
                if np.linalg.norm(template_mean) > 0 and np.linalg.norm(live_mean) > 0:
                    cos_sim = np.dot(template_mean, live_mean) / (np.linalg.norm(template_mean) * np.linalg.norm(live_mean))
                    mean_score = max(0, cos_sim)
                else:
                    mean_score = 0
                
                # Comparer les écarts-types
                template_std = np.array(template_hand["std"])
                live_std = np.array(live_hand["std"])
                std_diff = np.mean(np.abs(template_std - live_std))
                std_score = max(0, 1 - std_diff)
                
                # Comparer les keyframes (corrélation temporelle)
                template_keyframes = np.array(template_hand["keyframes"])
                live_keyframes = np.array(live_hand["keyframes"])
                
                keyframe_score = 0
                if len(template_keyframes) > 0 and len(live_keyframes) > 0:
                    # Rééchantillonner pour avoir le même nombre de keyframes
                    min_len = min(len(template_keyframes), len(live_keyframes))
                    if min_len > 0:
                        template_kf = template_keyframes[:min_len]
                        live_kf = live_keyframes[:min_len]
                        
                        # Distance moyenne entre keyframes
                        kf_diff = np.mean(np.abs(template_kf - live_kf))
                        keyframe_score = max(0, 1 - kf_diff)
                
                # Score combiné pour cette main
                hand_score = mean_score * 0.5 + std_score * 0.2 + keyframe_score * 0.3
                scores.append(hand_score)
                weights.append(1.0)
        
        # Comparer le visage (poids faible)
        if template_sig.get("face") and live_sig.get("face"):
            template_face_mean = np.array(template_sig["face"]["mean"])
            live_face_mean = np.array(live_sig["face"]["mean"])
            
            if len(template_face_mean) > 0 and len(live_face_mean) > 0:
                face_diff = np.mean(np.abs(template_face_mean - live_face_mean))
                face_score = max(0, 1 - face_diff)
                scores.append(face_score * 0.3)  # Poids réduit pour le visage
                weights.append(0.3)
        
        # Calculer le score final pondéré
        if scores:
            final_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Bonus si les deux mains sont utilisées de la même façon
            template_hands_used = sum(1 for h in template_sig["hands"].values() if h is not None)
            live_hands_used = sum(1 for h in live_sig["hands"].values() if h is not None)
            if template_hands_used == live_hands_used and template_hands_used > 0:
                final_score = min(1.0, final_score * 1.1)
            
            return final_score
        
        return 0.0
    
    def start_recognition(self):
        self.recognizing = True
        self.recognition_data = []
        self.recognition_start = time.time()
        self.result = None
        self.result_score = 0
        print("\n🔍 RECONNAISSANCE - 3 secondes")
    
    def stop_recognition(self):
        self.recognizing = False
        
        if len(self.recognition_data) < 10:
            print("❌ Pas assez de donnees capturees")
            return
        
        print(f"\n📊 Analyse de {len(self.recognition_data)} frames...")
        
        # Calculer la signature gestuelle en temps réel
        live_signature = self.compute_live_signature(self.recognition_data)
        
        if not live_signature:
            print("❌ Impossible de calculer la signature du geste")
            return
        
        # Comparer avec tous les gestes
        best_gesture = None
        best_score = 0
        all_scores = []
        
        for name, template_sig in self.templates.items():
            score = self.compare_signatures(template_sig, live_signature)
            all_scores.append((name, score))
            print(f"   {name}: {score:.2%}")
            
            if score > best_score:
                best_score = score
                best_gesture = name
        
        # Seuil de confiance (ajustable)
        confidence_threshold = 0.55
        
        # Lissage des résultats (évite les changements brusques)
        self.result_history.append((best_gesture, best_score))
        
        # Prendre le résultat le plus fréquent dans l'historique
        if len(self.result_history) > 0:
            from collections import Counter
            gesture_counts = Counter([r[0] for r in self.result_history if r[1] > confidence_threshold])
            if gesture_counts:
                most_common = gesture_counts.most_common(1)[0]
                if most_common[1] >= min(3, len(self.result_history)):
                    self.result = most_common[0]
                    self.result_score = best_score
                else:
                    self.result = best_gesture if best_score > confidence_threshold else None
                    self.result_score = best_score
            else:
                self.result = best_gesture if best_score > confidence_threshold else None
                self.result_score = best_score
        
        if self.result and self.result_score > confidence_threshold:
            print(f"\n🎯 RESULTAT: {self.result} (confiance: {self.result_score:.0%})")
        else:
            print(f"\n❌ Aucun geste reconnu")
            if best_gesture and best_score > 0.3:
                print(f"   Geste le plus proche: {best_gesture} ({best_score:.0%})")
    
    def draw_interface(self, frame):
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 255, 0), 2)
        
        cv2.putText(frame, "RECONNAISSANCE ASL", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "ESPACE: Reconnaitre   ESC: Quitter", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if self.recognizing:
            remaining = self.recognition_duration - (time.time() - self.recognition_start)
            bar_w = int((w - 40) * (remaining / self.recognition_duration))
            cv2.rectangle(frame, (20, h-30), (w-20, h-10), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h-30), (20 + bar_w, h-10), (0, 255, 0), -1)
            cv2.putText(frame, f"RECOGNITION... {int(remaining)}s", (20, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if self.result and not self.recognizing:
            cv2.rectangle(frame, (w//2-200, h//2-60), (w//2+200, h//2+60), (0, 0, 0), -1)
            cv2.rectangle(frame, (w//2-200, h//2-60), (w//2+200, h//2+60), (0, 255, 0), 3)
            cv2.putText(frame, self.result.upper(), (w//2-100, h//2-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
            cv2.putText(frame, f"Confiance: {self.result_score:.0%}", (w//2-80, h//2+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def draw_landmarks(self, frame, hand_result, face_result):
        h, w = frame.shape[:2]
        
        if hand_result and hand_result.multi_hand_landmarks:
            for hand_lms in hand_result.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        if face_result and face_result.multi_face_landmarks:
            for face_lms in face_result.multi_face_landmarks:
                for lm in face_lms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
        
        return frame
    
    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = self.hands.process(rgb)
            face_result = self.face_mesh.process(rgb)
            
            if self.recognizing:
                frame_data = {
                    "timestamp": time.time() - self.recognition_start,
                    "hands": {},
                    "face": None
                }
                
                if hand_result and hand_result.multi_hand_landmarks:
                    for handedness, hand_lms in zip(hand_result.multi_handedness, hand_result.multi_hand_landmarks):
                        label = handedness.classification[0].label
                        frame_data["hands"][label] = self.get_hand_data(hand_lms)
                
                if face_result and face_result.multi_face_landmarks:
                    frame_data["face"] = self.get_face_data(face_result.multi_face_landmarks[0])
                
                self.recognition_data.append(frame_data)
                
                if time.time() - self.recognition_start >= self.recognition_duration:
                    self.stop_recognition()
            
            frame = self.draw_landmarks(frame, hand_result, face_result)
            frame = self.draw_interface(frame)
            
            cv2.imshow("Gesture Recognizer", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == 32:
                if not self.recognizing:
                    self.start_recognition()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Termine!")

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.run()