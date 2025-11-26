"""
Módulo de Desenho e Visualização
Funções para desenhar vagas, veículos e informações no vídeo
"""
import cv2


# Cores em BGR
COLOR_FREE = (0, 255, 0)      # Verde - vaga livre
COLOR_OCCUPIED = (0, 0, 255)  # Vermelho - vaga ocupada
COLOR_TEXT = (255, 255, 255)  # Branco - texto
COLOR_PANEL_BG = (50, 50, 50) # Cinza escuro - fundo do painel


def draw_parking_spot(frame, coords, is_occupied, spot_id=None):
    """
    Desenha uma vaga de estacionamento no frame
    
    Args:
        frame: Frame onde desenhar
        coords: Coordenadas [x1, y1, x2, y2]
        is_occupied: True se ocupada, False se livre
        spot_id: ID da vaga (opcional)
    """
    x1, y1, x2, y2 = map(int, coords)
    
    # Define cor baseado no status
    color = COLOR_OCCUPIED if is_occupied else COLOR_FREE
    
    # Desenha retângulo da vaga
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Adiciona label com status
    status_text = "OCUPADA" if is_occupied else "LIVRE"
    if spot_id is not None:
        label = f"#{spot_id} - {status_text}"
    else:
        label = status_text
    
    # Fundo semi-transparente para o texto
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    
    # Posiciona label acima da vaga
    label_y = max(y1 - 10, text_height + 5)
    cv2.rectangle(
        frame,
        (x1, label_y - text_height - 5),
        (x1 + text_width + 5, label_y + baseline),
        color,
        -1
    )
    
    cv2.putText(
        frame,
        label,
        (x1 + 2, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),  # Texto preto
        2
    )


def draw_info_panel(frame, total_spots, free_spots, occupied_spots):
    """
    Desenha painel de informações no topo do frame
    
    Args:
        frame: Frame onde desenhar
        total_spots: Total de vagas
        free_spots: Vagas livres
        occupied_spots: Vagas ocupadas
    """
    h, w = frame.shape[:2]
    
    # Painel no topo
    panel_height = 80
    cv2.rectangle(frame, (0, 0), (w, panel_height), COLOR_PANEL_BG, -1)
    
    # Título
    cv2.putText(
        frame,
        "ESTACIONAMENTO VISION",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_TEXT,
        2
    )
    
    # Estatísticas
    stats_y = 60
    
    # Total
    cv2.putText(
        frame,
        f"Total: {total_spots}",
        (10, stats_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_TEXT,
        2
    )
    
    # Livres (verde)
    cv2.putText(
        frame,
        f"Livres: {free_spots}",
        (150, stats_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_FREE,
        2
    )
    
    # Ocupadas (vermelho)
    cv2.putText(
        frame,
        f"Ocupadas: {occupied_spots}",
        (310, stats_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLOR_OCCUPIED,
        2
    )
    
    # Taxa de ocupação
    if total_spots > 0:
        occupancy_rate = (occupied_spots / total_spots) * 100
        cv2.putText(
            frame,
            f"Taxa: {occupancy_rate:.1f}%",
            (w - 180, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2
        )


def draw_vehicle_box(frame, coords, vehicle_class, confidence):
    """
    Desenha caixa ao redor de um veículo detectado
    
    Args:
        frame: Frame onde desenhar
        coords: Coordenadas [x1, y1, x2, y2]
        vehicle_class: Tipo do veículo (car, motorcycle, etc)
        confidence: Confiança da detecção (0.0 - 1.0)
    """
    x1, y1, x2, y2 = map(int, coords)
    
    # Cor azul para veículos
    color = (255, 0, 0)
    
    # Desenha retângulo
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Label com classe e confiança
    label = f"{vehicle_class} {confidence:.2f}"
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def draw_center_cross(frame, coords, color=(0, 255, 255)):
    """
    Desenha uma cruz no centro de uma região (útil para debug)
    
    Args:
        frame: Frame onde desenhar
        coords: Coordenadas [x1, y1, x2, y2]
        color: Cor da cruz em BGR
    """
    x1, y1, x2, y2 = map(int, coords)
    
    # Calcula centro
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Tamanho da cruz
    size = 10
    
    # Desenha cruz
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)
    cv2.circle(frame, (cx, cy), 3, color, -1)
