"""
Sistema de Detecção de Vagas - Câmera IP
Detecta veículos automaticamente sem overlay de vagas
"""
import cv2
from pathlib import Path
from detector import VehicleDetector

# ========== CONFIGURAÇÃO ==========
# URL da câmera IP
CAMERA_URL = "http://192.168.1.5:8080/video"

# Modelo
USE_CUSTOM_MODEL = True
CUSTOM_MODEL_PATH = "car_detection_test/train/weights/best.pt"
PRETRAINED_MODEL = "yolov8n.pt"

# Modo de visualização
SHOW_BOXES = True  # Mostrar caixas ao redor dos veículos
SHOW_YOLO_LABELS = True  # Mostrar labels do YOLO
# ==================================


def main():
    print("=" * 70)
    print("📹 DETECÇÃO AUTOMÁTICA DE VEÍCULOS - CÂMERA IP")
    print("=" * 70)
    
    # Seleciona modelo
    if USE_CUSTOM_MODEL and Path(CUSTOM_MODEL_PATH).exists():
        model_path = CUSTOM_MODEL_PATH
        print(f"🎯 Modelo: Customizado ({model_path})")
    else:
        model_path = PRETRAINED_MODEL
        print(f"🔧 Modelo: Pré-treinado ({model_path})")
    
    # Inicializa detector
    print("\n🔄 Carregando detector...")
    detector = VehicleDetector(model_path=model_path)
    
    # Conecta à câmera IP
    print(f"\n📹 Conectando à câmera: {CAMERA_URL}")
    print("⏳ Aguarde...")
    
    # Desabilita verificação SSL para HTTPS
    import os
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    
    cap = cv2.VideoCapture(CAMERA_URL)
    
    # Tenta configurações alternativas se não conectar
    if not cap.isOpened():
        print("\n⚠️  Tentando sem SSL...")
        # Remove 's' do https
        alt_url = CAMERA_URL.replace('https://', 'http://')
        cap = cv2.VideoCapture(alt_url)
    
    if not cap.isOpened():
        print("\n❌ Erro ao conectar à câmera IP!")
        print("\nVerifique:")
        print("1. A URL está correta")
        print("2. A câmera está acessível na rede")
        print("3. Não há firewall bloqueando")
        print("4. O formato do stream é compatível (MJPEG/H264)")
        print("\n💡 Dica: Teste a URL no navegador primeiro")
        return
    
    print("✅ Conectado à câmera!")
    print("\n🚀 Sistema iniciado!")
    print("📝 Controles:")
    print("   ESC - Sair")
    print("   ESPAÇO - Pausar/Continuar")
    print("   S - Salvar screenshot")
    print("   B - Alternar caixas de detecção")
    
    frame_count = 0
    paused = False
    show_boxes = SHOW_BOXES
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n⚠️  Erro ao capturar frame ou fim do stream")
                break
            
            frame_count += 1
            
            # Detecta veículos
            vehicles = detector.detect_vehicles(frame)
            
            # Desenha detecções se habilitado
            if show_boxes:
                for vehicle in vehicles:
                    x1, y1, x2, y2 = map(int, vehicle['coords'])
                    conf = vehicle['confidence']
                    v_class = vehicle['class']
                    
                    # Cor baseada na confiança
                    if conf > 0.7:
                        color = (0, 255, 0)  # Verde
                    elif conf > 0.5:
                        color = (0, 255, 255)  # Amarelo
                    else:
                        color = (0, 165, 255)  # Laranja
                    
                    # Desenha retângulo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if SHOW_YOLO_LABELS:
                        label = f"Veiculo {conf:.0%}"
                        cv2.putText(
                            frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )
            
            # Painel de informações
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
            
            cv2.putText(
                frame, "DETECCAO DE VEICULOS",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            cv2.putText(
                frame, f"Veiculos: {len(vehicles)} | Frame: {frame_count}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            current_frame = frame.copy()
        
        # Mostra resultado
        cv2.imshow("Detecção de Veículos - Câmera IP", current_frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # ESPAÇO
            paused = not paused
            status = "⏸️  PAUSADO" if paused else "▶️  RODANDO"
            print(status)
        elif key == ord('b') or key == ord('B'):
            show_boxes = not show_boxes
            status = "ATIVADAS" if show_boxes else "DESATIVADAS"
            print(f"📦 Caixas de detecção: {status}")
        elif key == ord('s') or key == ord('S'):
            # Salva screenshot
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, current_frame)
            print(f"📸 Screenshot salvo: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Sistema encerrado")


if __name__ == "__main__":
    main()
