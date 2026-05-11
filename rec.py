import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import json
import time
import numpy as np

class GestureRecorder:
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
        
        # Liste des mots à enregistrer (votre liste originale)
        self.gestures = [
            "hello", "bye", "yes", "no",
            "stop", "please", "help", "wait", "work",
            "mom", "dad", "school", "friend", "family", "name",
            "thank you", "sorry", "want", "need", "go", "come",
            "see", "know", "think", "like", "eat", "drink", 
            "I(me)", "we", "you", "to",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
            "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
            "Love", "Food", "money", "home"
        ]
        
        # Dossier de stockage
        self.data_dir = "gesture_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Charger les gestes déjà enregistrés
        self.existing_gestures = self.load_existing_gestures()
        
        # État
        self.recording = False
        self.record_data = []
        self.current_index = 0
        self.record_start = 0
        self.record_duration = 5.0  # 5 SECONDES pour les gestes longs
        
        # Sauter les gestes déjà enregistrés
        self.skip_existing_gestures()
        
        print("="*50)
        print(" ENREGISTREMENT DES GESTES (5 secondes)")
        print("="*50)
        print(f"1er geste: {self.gestures[self.current_index]}")
        print(f"Gestes déjà enregistrés: {len(self.existing_gestures)}")
        print("\nCOMMANDES:")
        print("   R : Commencer l'enregistrement (5 secondes)")
        print("   N : Passer au geste suivant (sauvegarde auto)")
        print("   ← : Geste précédent")
        print("   → : Geste suivant")
        print("   ESC : Quitter")
        print("="*50)
    
    def load_existing_gestures(self):
        """Charge la liste des gestes déjà enregistrés"""
        existing = set()
        if os.path.exists(self.data_dir):
            for f in os.listdir(self.data_dir):
                if f.endswith('_sig.json'):
                    gesture_name = f.replace('_sig.json', '')
                    existing.add(gesture_name)
        return existing
    
    def skip_existing_gestures(self):
        """Saute les gestes déjà enregistrés"""
        while self.current_index < len(self.gestures):
            if self.gestures[self.current_index] not in self.existing_gestures:
                return
            self.current_index += 1
        
        if self.current_index >= len(self.gestures):
            print("\n🎉 TOUS LES GESTES SONT DÉJÀ ENREGISTRÉS !")
    
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
        
        # 1. Angles des doigts
        angles = self.compute_hand_angles(hand_points)
        features.extend(angles)
        
        # 2. Distances relatives
        distances = self.compute_hand_distances(hand_points)
        features.extend(distances)
        
        # 3. Position relative des bouts de doigts
        fingertips = [4, 8, 12, 16, 20]
        palm_base = np.array(hand_points[0])
        
        for tip in fingertips:
            if tip < len(hand_points):
                tip_pos = np.array(hand_points[tip])
                relative_pos = tip_pos - palm_base
                features.extend([relative_pos[0], relative_pos[1], relative_pos[2]])
        
        # 4. Courbure des doigts
        finger_bases = [1, 5, 9, 13, 17]
        for base, tip in zip(finger_bases, fingertips):
            if base < len(hand_points) and tip < len(hand_points):
                base_pos = np.array(hand_points[base])
                tip_pos = np.array(hand_points[tip])
                finger_len = np.linalg.norm(tip_pos - base_pos)
                features.append(finger_len)
        
        return features
    
    def compute_gesture_signature(self):
        """Calcule une signature unique pour le geste enregistré"""
        if not self.record_data:
            return None
        
        hand_features = {"Right": [], "Left": []}
        face_features = []
        
        for frame_data in self.record_data:
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
            "frames_original": len(self.record_data),
            "hands": {}
        }
        
        for hand_type in ["Right", "Left"]:
            if hand_features[hand_type]:
                features_array = np.array(hand_features[hand_type])
                signature["hands"][hand_type] = {
                    "mean": features_array.mean(axis=0).tolist(),
                    "std": features_array.std(axis=0).tolist(),
                    "num_frames": len(features_array),
                    "keyframes": features_array[::max(1, len(features_array)//10)].tolist()
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
    
    def save_current_gesture(self):
        """Sauvegarde le geste actuel"""
        gesture_name = self.gestures[self.current_index]
        
        if len(self.record_data) < 10:
            print(f"⚠️ Pas assez de données pour '{gesture_name}' ({len(self.record_data)} frames) - min 10 requises")
            return False
        
        signature = self.compute_gesture_signature()
        
        if not signature:
            print(f"❌ Erreur lors du calcul de la signature pour '{gesture_name}'")
            return False
        
        filename = os.path.join(self.data_dir, f"{gesture_name}_sig.json")
        
        data = {
            "gesture": gesture_name,
            "num_frames": len(self.record_data),
            "duration": self.record_duration,
            "signature": signature
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.existing_gestures.add(gesture_name)
        print(f"💾 Sauvegardé: {gesture_name} ({len(self.record_data)} frames)")
        return True
    
    def start_recording(self):
        if self.recording:
            return
        self.recording = True
        self.record_data = []
        self.record_start = time.time()
        print(f"\n🔴 ENREGISTREMENT DE '{self.gestures[self.current_index]}' - {self.record_duration} secondes")
    
    def stop_recording(self):
        self.recording = False
        print(f"   Enregistrement terminé: {len(self.record_data)} frames")
        if len(self.record_data) < 10:
            print(f"   ⚠️ Pas assez de frames ({len(self.record_data)}/10) - Recommencez avec R")
    
    def next_gesture(self):
        """Passe au geste suivant avec sauvegarde automatique"""
        if self.recording:
            print("⚠️ Attendez la fin de l'enregistrement")
            return True
        
        # Sauvegarder le geste actuel si des données existent
        if self.record_data:
            if len(self.record_data) >= 10:
                self.save_current_gesture()
            else:
                print(f"⚠️ Geste '{self.gestures[self.current_index]}' ignoré: trop peu de frames ({len(self.record_data)}/10)")
            self.record_data = []
        
        # Passer au suivant
        self.current_index += 1
        
        # Sauter les gestes déjà enregistrés
        while self.current_index < len(self.gestures):
            if self.gestures[self.current_index] not in self.existing_gestures:
                break
            print(f"⏭️ Geste '{self.gestures[self.current_index]}' déjà enregistré - ignoré")
            self.current_index += 1
        
        if self.current_index >= len(self.gestures):
            print("\n🎉 TOUS LES GESTES ONT ÉTÉ ENREGISTRÉS !")
            return False
        
        print(f"\n📝 Prochain geste: {self.gestures[self.current_index]} ({self.current_index + 1}/{len(self.gestures)})")
        return True
    
    def prev_gesture(self):
        """Retourne au geste précédent"""
        if self.recording:
            print("⚠️ Attendez la fin de l'enregistrement")
            return
        
        if self.current_index > 0:
            self.current_index -= 1
            self.record_data = []
            print(f"\n⬅️ Retour au geste: {self.gestures[self.current_index]} ({self.current_index + 1}/{len(self.gestures)})")
    
    def draw_interface(self, frame):
        h, w = frame.shape[:2]
        
        # Barre du haut
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 255, 0), 2)
        
        # Geste actuel
        gesture = self.gestures[self.current_index]
        cv2.putText(frame, f"GESTE: {gesture.upper()}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progression
        prog = f"PROGRESSION: {self.current_index + 1}/{len(self.gestures)}"
        cv2.putText(frame, prog, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Gestes enregistrés
        recorded_text = f"ENREGISTRES: {len(self.existing_gestures)}"
        cv2.putText(frame, recorded_text, (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Zone des commandes (à droite)
        cv2.rectangle(frame, (w-180, 10), (w-10, 95), (0, 0, 0), -1)
        cv2.rectangle(frame, (w-180, 10), (w-10, 95), (100, 100, 100), 1)
        cv2.putText(frame, "R", (w-170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Enregistrer", (w-150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "N", (w-170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Next/Save", (w-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "← →", (w-170, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Changer", (w-150, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Zone d'enregistrement
        if self.recording:
            elapsed = time.time() - self.record_start
            remaining = max(0, self.record_duration - elapsed)
            progress = min(100, (elapsed / self.record_duration) * 100)
            
            # Barre de progression en bas
            bar_w = int((w - 40) * (progress / 100))
            cv2.rectangle(frame, (20, h-35), (w-20, h-15), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h-35), (20 + bar_w, h-15), (0, 0, 255), -1)
            
            # Texte d'enregistrement
            cv2.putText(frame, f"🔴 RECORDING... {remaining:.1f}s restantes", (20, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Frames: {len(self.record_data)}", (20, h-55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Point rouge clignotant
            if int(time.time() * 2) % 2:
                cv2.circle(frame, (w-30, 30), 10, (0, 0, 255), -1)
        
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
            
            # Enregistrement
            if self.recording:
                frame_data = {
                    "timestamp": time.time() - self.record_start,
                    "hands": {},
                    "face": None
                }
                
                if hand_result and hand_result.multi_hand_landmarks:
                    for handedness, hand_lms in zip(hand_result.multi_handedness, hand_result.multi_hand_landmarks):
                        label = handedness.classification[0].label
                        frame_data["hands"][label] = self.get_hand_data(hand_lms)
                
                if face_result and face_result.multi_face_landmarks:
                    frame_data["face"] = self.get_face_data(face_result.multi_face_landmarks[0])
                
                self.record_data.append(frame_data)
                
                # Fin automatique après la durée
                if time.time() - self.record_start >= self.record_duration:
                    self.stop_recording()
            
            # Dessiner les éléments
            frame = self.draw_landmarks(frame, hand_result, face_result)
            frame = self.draw_interface(frame)
            
            cv2.imshow("Gesture Recorder", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                # Sauvegarder le geste en cours avant de quitter
                if self.record_data and len(self.record_data) >= 10:
                    self.save_current_gesture()
                break
            elif key == ord('r') or key == ord('R'):  # R = Enregistrer
                if not self.recording:
                    self.start_recording()
                else:
                    print("⚠️ Enregistrement déjà en cours")
            elif key == ord('n') or key == ord('N'):  # N = Next
                if not self.next_gesture():
                    break
            elif key == 81 or key == 2424832:  # Flèche gauche
                self.prev_gesture()
            elif key == 83 or key == 2555904:  # Flèche droite
                if not self.recording:
                    if self.record_data and len(self.record_data) >= 10:
                        self.save_current_gesture()
                    self.record_data = []
                    self.current_index += 1
                    while self.current_index < len(self.gestures):
                        if self.gestures[self.current_index] not in self.existing_gestures:
                            break
                        self.current_index += 1
                    if self.current_index < len(self.gestures):
                        print(f"\n📝 Geste: {self.gestures[self.current_index]} ({self.current_index + 1}/{len(self.gestures)})")
                    else:
                        print("\n🎉 Tous les gestes sont enregistrés !")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Enregistrement terminé!")

if __name__ == "__main__":
    recorder = GestureRecorder()
    recorder.run()