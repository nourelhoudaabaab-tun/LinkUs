[app]

# ⭐ METS TON CHEMIN COMPLET ICI
source.dir = "C:/LinkUsApp"

title = LinkUs
package.name = linkus
package.domain = org.linkus
version = 1.0.0

requirements = python3,kivy,opencv-python,mediapipe,numpy

android.permissions = CAMERA
android.orientation = portrait
android.wakelock = True
icon.filename = C:/LinkUsApp/icon.png

fullscreen = 0
window.size = 400, 700

ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.ios_deploy_url = https://github.com/phonegap/ios-deploy
ios.codesign.allowed = False