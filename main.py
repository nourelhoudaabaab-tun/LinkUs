import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import time
import json
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.textinput import TextInput
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

from recognize import GestureRecognizer
from avatar import AvatarGenerator


class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=30, spacing=25)
        
        title = Label(
            text='ASL TRANSLATION',
            font_size='28sp',
            color=(0.2, 0.5, 0.8, 1),
            size_hint=(1, 0.25)
        )
        layout.add_widget(title)
        
        btn_signe_texte = Button(
            text='SIGN to TEXT',
            background_color=(0.2, 0.55, 0.9, 1),
            color=(1, 1, 1, 1),
            font_size='18sp',
            bold=True,
            size_hint=(1, 0.2)
        )
        btn_signe_texte.bind(on_press=self.go_to_recognition)
        layout.add_widget(btn_signe_texte)
        
        btn_texte_signe = Button(
            text='TEXT to SIGN',
            background_color=(0.3, 0.65, 0.95, 1),
            color=(1, 1, 1, 1),
            font_size='18sp',
            bold=True,
            size_hint=(1, 0.2)
        )
        btn_texte_signe.bind(on_press=self.go_to_text_to_sign)
        layout.add_widget(btn_texte_signe)
        
        btn_quit = Button(
            text='QUITTER',
            background_color=(0.6, 0.65, 0.8, 1),
            color=(1, 1, 1, 1),
            font_size='16sp',
            bold=True,
            size_hint=(1, 0.12)
        )
        btn_quit.bind(on_press=self.quit_app)
        layout.add_widget(btn_quit)
        
        self.add_widget(layout)
    
    def go_to_recognition(self, instance):
        self.manager.current = 'recognition'
    
    def go_to_text_to_sign(self, instance):
        self.manager.current = 'text_to_sign'
    
    def quit_app(self, instance):
        App.get_running_app().stop()


class RecognitionScreen(Screen):
    """Écran de reconnaissance - Version automatique (fin quand main disparaît)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recognizer = None
        self.video_event = None
        
        # Variables pour la phrase
        self.current_sentence = []
        
        # Variables pour la reconnaissance automatique
        self.is_capturing = False      # Est-ce qu'on capture un geste ?
        self.capture_data = []         # Données du geste en cours
        self.capture_start_time = 0
        self.max_capture_duration = 5.0  # Durée max de capture (5 secondes)
        self.last_recognition_time = 0
        self.cooldown = 0.8            # Délai entre deux reconnaissances
        
        self.build_ui()
    
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=15, spacing=10)
        
        # Header
        header = BoxLayout(size_hint=(1, 0.08), spacing=15)
        
        btn_back = Button(
            text='RETOUR',
            background_color=(0.5, 0.55, 0.7, 1),
            color=(1, 1, 1, 1),
            font_size='12sp',
            bold=True,
            size_hint_x=0.25
        )
        btn_back.bind(on_press=self.go_back)
        header.add_widget(btn_back)
        
        title = Label(
            text='SIGN to TEXT',
            font_size='16sp',
            color=(0.2, 0.5, 0.8, 1),
            size_hint_x=0.5
        )
        header.add_widget(title)
        
        btn_clear = Button(
            text='EFFACER',
            background_color=(0.7, 0.5, 0.2, 1),
            color=(1, 1, 1, 1),
            font_size='11sp',
            bold=True,
            size_hint_x=0.25
        )
        btn_clear.bind(on_press=self.clear_sentence)
        header.add_widget(btn_clear)
        
        layout.add_widget(header)
        
        # Zone de phrase
        sentence_frame = BoxLayout(orientation='vertical', size_hint=(1, 0.18), padding=[10, 5])
        
        sentence_label_title = Label(
            text='PHRASE EN COURS:',
            font_size='12sp',
            color=(0.2, 0.5, 0.8, 1),
            size_hint=(1, 0.25)
        )
        sentence_frame.add_widget(sentence_label_title)
        
        self.sentence_display = Label(
            text='---',
            font_size='18sp',
            color=(0.1, 0.3, 0.5, 1),
            halign='center',
            valign='middle',
            size_hint=(1, 0.75)
        )
        self.sentence_display.bind(size=self.sentence_display.setter('text_size'))
        sentence_frame.add_widget(self.sentence_display)
        
        layout.add_widget(sentence_frame)
        
        # Zone vidéo
        self.video_widget = Image(
            size_hint=(1, 0.4),
            keep_ratio=True,
            allow_stretch=True
        )
        layout.add_widget(self.video_widget)
        
        # Dernier mot reconnu
        self.result_label = Label(
            text='DERNIER MOT\n---',
            font_size='16sp',
            color=(0.3, 0.55, 0.85, 1),
            halign='center',
            valign='middle',
            size_hint=(1, 0.08)
        )
        layout.add_widget(self.result_label)
        
        # État (instructions)
        self.status_label = Label(
            text='👋 Montrez votre MAIN pour commencer - RETIREZ votre main pour valider',
            font_size='11sp',
            color=(0.2, 0.6, 0.3, 1),
            size_hint=(1, 0.05)
        )
        layout.add_widget(self.status_label)
        
        # Barre de progression
        self.progress_bar = ProgressBar(max=100, value=0, size_hint=(1, 0.02))
        layout.add_widget(self.progress_bar)
        
        # Boutons
        buttons = BoxLayout(size_hint=(1, 0.08), spacing=10)
        
        self.btn_send = Button(
            text='VALIDER PHRASE',
            background_color=(0.3, 0.65, 0.3, 1),
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True
        )
        self.btn_send.bind(on_press=self.send_sentence)
        buttons.add_widget(self.btn_send)
        
        layout.add_widget(buttons)
        
        # Infos
        self.stats_label = Label(
            text='Gestes disponibles: 0',
            font_size='10sp',
            color=(0.3, 0.5, 0.7, 1),
            size_hint=(1, 0.04)
        )
        layout.add_widget(self.stats_label)
        
        self.add_widget(layout)
    
    def detect_hand_present(self, hand_result):
        """Détecte si une main est présente"""
        return hand_result and hand_result.multi_hand_landmarks and len(hand_result.multi_hand_landmarks) > 0
    
    def clear_sentence(self, instance):
        self.current_sentence = []
        self.sentence_display.text = '---'
        self.status_label.text = '✏️ Phrase effacée'
        Clock.schedule_once(lambda dt: self.reset_status(), 2)
    
    def update_sentence_display(self):
        if self.current_sentence:
            self.sentence_display.text = ' '.join(self.current_sentence)
        else:
            self.sentence_display.text = '---'
    
    def add_word_to_sentence(self, word):
        self.current_sentence.append(word)
        self.update_sentence_display()
        self.result_label.text = f'DERNIER MOT\n{word.upper()}'
        self.result_label.color = (0.2, 0.6, 0.3, 1)
        
        # Réinitialiser l'affichage après 1.5 secondes
        Clock.schedule_once(lambda dt: self.reset_result(), 1.5)
    
    def reset_result(self):
        if not self.is_capturing:
            self.result_label.text = 'DERNIER MOT\n---'
            self.result_label.color = (0.3, 0.55, 0.85, 1)
    
    def reset_status(self):
        if not self.is_capturing:
            self.status_label.text = '👋 Montrez votre MAIN - RETIREZ votre main pour valider le geste'
            self.status_label.color = (0.2, 0.6, 0.3, 1)
    
    def start_capture(self):
        """Démarre la capture du geste (quand une main apparaît)"""
        if self.is_capturing:
            return
        
        current_time = time.time()
        if current_time - self.last_recognition_time < self.cooldown:
            return
        
        self.is_capturing = True
        self.capture_data = []
        self.capture_start_time = current_time
        
        self.status_label.text = '🎬 CAPTURE EN COURS... Faites votre geste, puis retirez la main'
        self.status_label.color = (0.8, 0.5, 0.2, 1)
        self.progress_bar.value = 0
    
    def stop_capture_and_recognize(self):
        """Arrête la capture et lance la reconnaissance (quand la main disparaît)"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        # Vérifier si on a assez de données
        if len(self.capture_data) < 8:
            self.status_label.text = '❌ Geste trop court - Recommencez (maintenez la main plus longtemps)'
            self.status_label.color = (0.7, 0.2, 0.2, 1)
            self.progress_bar.value = 0
            Clock.schedule_once(lambda dt: self.reset_status(), 2.5)
            return
        
        # Lancer la reconnaissance
        self.recognize_gesture()
    
    def recognize_gesture(self):
        """Reconnaît le geste à partir des données capturées"""
        self.status_label.text = '🔄 Analyse du geste...'
        
        if not self.recognizer or not self.capture_data:
            self.status_label.text = '❌ Erreur de reconnaissance'
            return
        
        # Calculer la signature live
        live_signature = self.compute_signature_from_data(self.capture_data)
        
        if not live_signature:
            self.status_label.text = '❌ Impossible de calculer la signature'
            return
        
        # Comparer avec les templates
        best_gesture = None
        best_score = 0
        
        for name, template in self.recognizer.templates.items():
            score = self.recognizer.compare_signatures(template, live_signature)
            if score > best_score:
                best_score = score
                best_gesture = name
        
        # Seuil de confiance
        if best_gesture and best_score > 0.4:
            self.last_recognition_time = time.time()
            self.add_word_to_sentence(best_gesture)
            self.status_label.text = f'✓ {best_gesture} reconnu ! ({best_score:.0%})'
            self.status_label.color = (0.2, 0.6, 0.3, 1)
            self.progress_bar.value = 0
            
            Clock.schedule_once(lambda dt: self.reset_status(), 2)
        else:
            self.status_label.text = f'❌ Geste non reconnu (score: {best_score:.0%})'
            self.status_label.color = (0.7, 0.2, 0.2, 1)
            self.progress_bar.value = 0
            
            if best_gesture:
                self.status_label.text += f' - Le plus proche: {best_gesture}'
            
            Clock.schedule_once(lambda dt: self.reset_status(), 2.5)
    
    def compute_signature_from_data(self, frames_data):
        """Calcule une signature à partir des données capturées"""
        if not frames_data or len(frames_data) < 5:
            return None
        
        hand_features = {"Right": [], "Left": []}
        
        for frame_data in frames_data:
            for hand_type in ["Right", "Left"]:
                hand_points = frame_data.get("hands", {}).get(hand_type)
                if hand_points and self.recognizer:
                    features = self.recognizer.compute_hand_features(hand_points)
                    if features:
                        hand_features[hand_type].append(features)
        
        signature = {"hands": {}}
        
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
        
        signature["face"] = None
        return signature
    
    def send_sentence(self, instance):
        if self.current_sentence:
            sentence = ' '.join(self.current_sentence)
            self.show_sentence_popup(sentence)
        else:
            self.status_label.text = '❌ Aucune phrase à valider'
            Clock.schedule_once(lambda dt: self.reset_status(), 1.5)
    
    def show_sentence_popup(self, sentence):
        content = BoxLayout(orientation='vertical', padding=15, spacing=15)
        
        sentence_label = Label(
            text=f'PHRASE:\n\n"{sentence.upper()}"',
            font_size='16sp',
            color=(0.2, 0.5, 0.8, 1),
            halign='center',
            size_hint_y=0.8
        )
        sentence_label.bind(size=sentence_label.setter('text_size'))
        content.add_widget(sentence_label)
        
        btn_layout = BoxLayout(size_hint_y=0.2, spacing=10)
        
        copy_btn = Button(
            text='📋 COPIER',
            background_color=(0.3, 0.6, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        copy_btn.bind(on_press=lambda x: self.copy_to_clipboard(sentence))
        btn_layout.add_widget(copy_btn)
        
        ok_btn = Button(
            text='OK',
            background_color=(0.2, 0.55, 0.9, 1),
            color=(1, 1, 1, 1)
        )
        btn_layout.add_widget(ok_btn)
        content.add_widget(btn_layout)
        
        popup = Popup(
            title='PHRASE VALIDÉE',
            title_color=(0.2, 0.5, 0.8, 1),
            content=content,
            size_hint=(0.8, 0.5)
        )
        ok_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def copy_to_clipboard(self, text):
        from kivy.core.clipboard import Clipboard
        Clipboard.copy(text)
        self.status_label.text = '📋 Phrase copiée dans le presse-papier'
    
    def update_video(self, dt):
        if not self.recognizer or not self.recognizer.cap:
            return
        
        try:
            ret, frame = self.recognizer.cap.read()
            if not ret:
                return
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                hand_result = self.recognizer.hands.process(rgb)
            except:
                hand_result = None
            
            # === LOGIQUE AUTOMATIQUE MODIFIÉE ===
            hand_present = self.detect_hand_present(hand_result)
            
            # Variable pour suivre l'état précédent (nécessite un attribut persistant)
            if not hasattr(self, 'hand_was_present'):
                self.hand_was_present = False
            
            # DÉTECTION : Main apparaît -> Démarrer la capture
            if hand_present and not self.hand_was_present and not self.is_capturing:
                # La main vient d'apparaître
                current_time = time.time()
                if current_time - self.last_recognition_time >= self.cooldown:
                    self.start_capture()
            
            # DÉTECTION : Main disparaît -> Terminer la capture et reconnaître
            if not hand_present and self.hand_was_present and self.is_capturing:
                # La main vient de disparaître
                self.stop_capture_and_recognize()
            
            # Mise à jour de l'état précédent
            self.hand_was_present = hand_present
            
            # Gestion de la durée maximale de capture (sécurité)
            if self.is_capturing:
                elapsed = time.time() - self.capture_start_time
                if elapsed > self.max_capture_duration:
                    # Durée max atteinte, forcer la fin
                    self.stop_capture_and_recognize()
            
            # Si on capture un geste, enregistrer les données
            if self.is_capturing:
                frame_data = {
                    "timestamp": time.time() - self.capture_start_time,
                    "hands": {},
                    "face": None
                }
                
                if hand_result and hand_result.multi_hand_landmarks:
                    for handedness, hand_lms in zip(hand_result.multi_handedness, hand_result.multi_hand_landmarks):
                        label = handedness.classification[0].label
                        frame_data["hands"][label] = self.recognizer.get_hand_data(hand_lms)
                
                self.capture_data.append(frame_data)
                
                # Mettre à jour la progression (basée sur le temps écoulé)
                elapsed = time.time() - self.capture_start_time
                progress = min(100, (elapsed / 3.0) * 100)  # 3 secondes = 100%
                self.progress_bar.value = progress
                
                # Afficher l'indicateur de capture sur la vidéo
                #h, w = frame.shape[:2]
                #cv2.rectangle(frame, (0, h-30), (int(w * progress / 100), h), (0, 255, 0), -1)
                #cv2.putText(frame, f"CAPTURE - {len(self.capture_data)} frames", (10, 60), 
                           #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
               # cv2.putText(frame, "RETIREZ LA MAIN POUR VALIDER", (10, 90), 
                          # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Afficher les instructions sur la vidéo
            h, w = frame.shape[:2]
            if hand_present and not self.is_capturing:
                cv2.putText(frame, "MAIN DETECTEE - GESTE ENREGISTRE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif not hand_present and not self.is_capturing:
                cv2.putText(frame, "AUCUNE MAIN - MONTREZ VOTRE GESTE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # Dessiner les points de la main
            #if hand_result and hand_result.multi_hand_landmarks:
                #h, w = frame.shape[:2]
                #for hand_lms in hand_result.multi_hand_landmarks:
                    #for lm in hand_lms.landmark:
                        #x, y = int(lm.x * w), int(lm.y * h)
                        #cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Convertir pour Kivy
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]),
                colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.video_widget.texture = texture
            
        except Exception as e:
            print(f"Erreur vidéo: {e}")
    
    def on_enter(self):
        print("Initialisation de la reconnaissance automatique...")
        try:
            if self.recognizer is None:
                self.recognizer = GestureRecognizer()
                self.stats_label.text = f'Gestes: {len(self.recognizer.templates)}'
                self.video_event = Clock.schedule_interval(self.update_video, 1.0 / 30.0)
                self.hand_was_present = False  # Initialiser l'état
                print("✅ Prêt - La reconnaissance se termine quand vous retirez la main")
        except Exception as e:
            print(f"Erreur: {e}")
            self.status_label.text = f'Erreur caméra'
    
    def on_leave(self):
        if self.video_event:
            self.video_event.cancel()
            self.video_event = None
        if self.recognizer and self.recognizer.cap:
            self.recognizer.cap.release()
    
    def go_back(self, instance):
        self.manager.current = 'menu'


class TextToSignScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avatar = None
        self.video_event = None
        self.video_cap = None
        self.build_ui()
    
    def build_ui(self):
        layout = BoxLayout(orientation='vertical', padding=15, spacing=12)
        
        header = BoxLayout(size_hint=(1, 0.08), spacing=15)
        
        btn_back = Button(
            text='RETOUR',
            background_color=(0.5, 0.55, 0.7, 1),
            color=(1, 1, 1, 1),
            font_size='12sp',
            bold=True,
            size_hint_x=0.3
        )
        btn_back.bind(on_press=self.go_back)
        header.add_widget(btn_back)
        
        title = Label(
            text='TEXT to SIGN',
            font_size='16sp',
            color=(0.2, 0.5, 0.8, 1),
            size_hint_x=0.7
        )
        header.add_widget(title)
        
        layout.add_widget(header)
        
        input_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10, padding=[10, 5])
        
        self.text_input = TextInput(
            hint_text="Entrez un mot (ex: hello, yes, mom, dad, please, stop)",
            multiline=False,
            font_size='14sp',
            size_hint_x=0.7
        )
        input_layout.add_widget(self.text_input)
        
        self.btn_translate = Button(
            text='TRADUIRE',
            background_color=(0.2, 0.55, 0.9, 1),
            color=(1, 1, 1, 1),
            font_size='14sp',
            bold=True,
            size_hint_x=0.3
        )
        self.btn_translate.bind(on_press=self.translate_text)
        input_layout.add_widget(self.btn_translate)
        
        layout.add_widget(input_layout)
        
        self.video_widget = Image(
            size_hint=(1, 0.55),
            keep_ratio=True,
            allow_stretch=True
        )
        layout.add_widget(self.video_widget)
        
        self.result_label = Label(
            text='Mot: ---',
            font_size='16sp',
            color=(0.3, 0.55, 0.85, 1),
            size_hint=(1, 0.08)
        )
        layout.add_widget(self.result_label)
        
        self.status_label = Label(
            text='Entrez un mot et appuyez sur TRADUIRE',
            font_size='11sp',
            color=(0.2, 0.6, 0.3, 1),
            size_hint=(1, 0.06)
        )
        layout.add_widget(self.status_label)
        
        self.add_widget(layout)
    
    def on_enter(self):
        try:
            self.avatar = AvatarGenerator(video_dir="video_blender")
            self.status_label.text = "Avatar prêt"
        except Exception as e:
            self.status_label.text = f"Erreur: {str(e)[:50]}"
    
    def on_leave(self):
        self.stop_video()
    
    def stop_video(self):
        if self.video_event:
            self.video_event.cancel()
            self.video_event = None
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.video_widget.texture = None
    
    def translate_text(self, instance):
        word = self.text_input.text.strip().lower()
        
        if not word:
            self.show_message("Veuillez entrer un mot", "error")
            return
        
        self.stop_video()
        
        self.result_label.text = f'Mot: {word.upper()}'
        self.status_label.text = f'Recherche du signe pour "{word}"...'
        self.btn_translate.disabled = True
        
        if self.avatar and self.avatar.get_video_path(word):
            video_path = self.avatar.get_video_path(word)
            self.status_label.text = f'Lecture du signe pour "{word}"'
            self.play_video_in_app(video_path)
        else:
            msg = f'Mot "{word}" non trouvé'
            if self.avatar:
                msg += f'\nMots: {", ".join(self.avatar.get_available_words())}'
            self.show_message(msg, "error")
            self.reset_ui()
    
    def play_video_in_app(self, video_path):
        try:
            self.video_cap = cv2.VideoCapture(video_path)
            
            if not self.video_cap.isOpened():
                self.show_message("Impossible d'ouvrir la vidéo", "error")
                self.reset_ui()
                return
            
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
            
            self.video_event = Clock.schedule_interval(lambda dt: self.update_video_frame(), delay)
            
        except Exception as e:
            self.show_message(f"Erreur: {str(e)}", "error")
            self.reset_ui()
    
    def update_video_frame(self):
        if not self.video_cap:
            self.stop_video()
            self.reset_ui()
            return
        
        ret, frame = self.video_cap.read()
        
        if not ret:
            self.stop_video()
            self.reset_ui()
            return
        
        try:
            h, w = frame.shape[:2]
            if h > 0 and w > 0:
                frame = cv2.flip(frame, 0)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                texture = Texture.create(size=(w, h), colorfmt='rgb')
                texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                self.video_widget.texture = texture
        except Exception as e:
            print(f"Erreur: {e}")
    
    def show_message(self, message, type_msg):
        content = BoxLayout(orientation='vertical', padding=15, spacing=15)
        color = (0.2, 0.6, 0.3, 1) if type_msg == "success" else (0.7, 0.2, 0.2, 1)
        
        msg_label = Label(
            text=message,
            font_size='13sp',
            color=color,
            halign='center',
            size_hint_y=0.8
        )
        content.add_widget(msg_label)
        
        ok_btn = Button(
            text='OK',
            background_color=(0.2, 0.55, 0.9, 1),
            color=(1, 1, 1, 1),
            size_hint_y=0.2
        )
        content.add_widget(ok_btn)
        
        popup = Popup(
            title='Information' if type_msg == "success" else 'Erreur',
            title_color=color,
            content=content,
            size_hint=(0.8, 0.4)
        )
        ok_btn.bind(on_press=popup.dismiss)
        Clock.schedule_once(lambda dt: popup.open(), 0)
    
    def reset_ui(self):
        self.btn_translate.disabled = False
        self.status_label.text = 'Prêt - Entrez un nouveau mot'
        self.status_label.color = (0.2, 0.6, 0.3, 1)
        self.text_input.text = ""
    
    def go_back(self, instance):
        self.stop_video()
        self.manager.current = 'menu'


class GestureApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "ASL Translation"
    
    def build(self):
        Window.clearcolor = (0.95, 0.95, 1, 1)
        Window.size = (400, 700)
        
        sm = ScreenManager()
        sm.add_widget(MenuScreen(name='menu'))
        sm.add_widget(RecognitionScreen(name='recognition'))
        sm.add_widget(TextToSignScreen(name='text_to_sign'))
        
        return sm


if __name__ == '__main__':
    GestureApp().run()