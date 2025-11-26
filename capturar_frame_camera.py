"""
Captura um frame da câmera IP para marcar as vagas
"""
import cv2
import sys

# URL da câmera IP
CAMERA_URL = "https://10.118.3.92:8080/video"

print("=" * 70)
print("📷 CAPTURA DE FRAME DA CÂMERA IP")
print("=" * 70)
print(f"\n🔗 Conectando em: {CAMERA_URL}")

# Tentar capturar
cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("\n❌ Não foi possível conectar à câmera!")
    print("\n💡 Tente também:")
    print("   - http://10.118.3.92:8080/video")
    print("   - rtsp://10.118.3.92:8080/h264_ulaw.sdp")
    sys.exit(1)

print("✅ Conectado!")
print("\n📸 Capturando frame...")

# Capturar frame
ret, frame = cap.read()

if not ret or frame is None:
    print("❌ Erro ao capturar frame")
    cap.release()
    sys.exit(1)

# Salvar frame
output_path = "camera_frame.jpg"
cv2.imwrite(output_path, frame)

print(f"✅ Frame salvo: {output_path}")
print(f"   Resolução: {frame.shape[1]}x{frame.shape[0]}")

# Mostrar preview por 3 segundos
print("\n👀 Mostrando preview (pressione qualquer tecla para continuar)...")
cv2.imshow("Frame Capturado", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap.release()

print("\n" + "=" * 70)
print("📋 PRÓXIMOS PASSOS:")
print("=" * 70)
print("\n1. Execute: python marcar_vagas.py camera_frame.jpg")
print("2. Marque as vagas clicando e arrastando")
print("3. Pressione 's' para salvar")
print("4. Execute: python sistema_vagas_cnn.py")
print("\n" + "=" * 70)
