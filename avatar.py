# avatar.py - Version adaptée pour Kivy
import os
import cv2

class AvatarGenerator:
    """Classe pour gérer l'avatar et les vidéos de signes"""
    
    def __init__(self, video_dir="video_blender"):
        """
        Initialise l'avatar avec le dossier contenant les vidéos
        
        Args:
            video_dir (str): Chemin vers le dossier contenant les vidéos
        """
        self.video_dir = video_dir
        
        # Dictionnaire mot -> nom du fichier vidéo
        self.videos = {
            "hello": "hello.mp4",
            "stop": "stop.mp4",
            "dad": "dad.mp4",
            "please": "please.mp4",
            "yes": "yes.mp4",
            "mom": "mom.mp4",
        }
    
    def get_video_path(self, word):
        """
        Retourne le chemin complet de la vidéo pour un mot
        
        Args:
            word (str): Le mot à traduire en signe
            
        Returns:
            str or None: Chemin de la vidéo ou None si non trouvé
        """
        word = word.lower().strip()
        
        if word not in self.videos:
            return None
        
        video_filename = self.videos[word]
        video_path = os.path.join(self.video_dir, video_filename)
        
        if os.path.exists(video_path):
            return video_path
        return None
    
    def get_available_words(self):
        """
        Retourne la liste des mots disponibles
        
        Returns:
            list: Liste des mots disponibles
        """
        return list(self.videos.keys())
    
    def video_exists(self, word):
        """
        Vérifie si une vidéo existe pour un mot
        
        Args:
            word (str): Le mot à vérifier
            
        Returns:
            bool: True si la vidéo existe, False sinon
        """
        return self.get_video_path(word) is not None