import pygame
import json
import numpy as np

# Initialize pygame
pygame.init()
width, height = 500, 500
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("trAIl")

# Colors
WHITE, BLACK, BLUE, GREEN, RED, GRAY = (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (200, 200, 200)

# UI Elements
clear_button = pygame.Rect(50, 450, 100, 40)
train_button = pygame.Rect(180, 450, 100, 40)
input_box = pygame.Rect(300, 450, 150, 40)
font = pygame.font.Font(None, 30)

# Doodle storage
drawing = False
points = []
memory_file = "memory.json"
text_input = ""
prediction_result = ""

# Load or initialize memory
try:
    with open(memory_file, "r") as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = {"doodles": []}
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=4)

def extract_features(points):
    """Extracts advanced features for intelligent shape prediction."""
    if not points:
        return None

    x_vals = np.array([p[0] for p in points])
    y_vals = np.array([p[1] for p in points])

    width = max(x_vals) - min(x_vals)
    height = max(y_vals) - min(y_vals)
    density = len(points) / (width * height + 1)
    curve_factor = np.std(y_vals) / (np.std(x_vals) + 1)
    aspect_ratio = width / (height + 1)
    centroid_x = np.mean(x_vals)
    centroid_y = np.mean(y_vals)

    return {
        "width": int(width),
        "height": int(height),
        "density": float(density),
        "curve_factor": float(curve_factor),
        "aspect_ratio": float(aspect_ratio),
        "centroid_x": int(centroid_x),
        "centroid_y": int(centroid_y)
    }

def similarity_score(features1, features2):
    """Calculates similarity using cosine similarity."""
    f1 = np.array(list(features1.values()))
    f2 = np.array(list(features2.values()))
    dot_product = np.dot(f1, f2)
    magnitude = np.linalg.norm(f1) * np.linalg.norm(f2)
    return dot_product / (magnitude + 1e-5)

def predict_doodle(points):
    """Predicts a shape intelligently while drawing."""
    features = extract_features(points)
    if features is None:
        return "No doodle detected"

    best_match = None
    best_score = -1  

    for prev_doodle in memory["doodles"]:
        prev_features = {key: prev_doodle.get(key, 0.0) for key in features.keys()}
        score = similarity_score(features, prev_features)
        if score > best_score:
            best_score = score
            best_match = prev_doodle["label"]

    return f"Predicted: {best_match} ({best_score:.2f})" if best_match else "Unknown Shape - Train Me!"

def train_doodle(points, label):
    """Trains the doodle predictor with labeled data."""
    features = extract_features(points)
    if features is None or not label:
        return

    memory["doodles"].append({**features, "label": label})
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=4)

def draw_ui():
    """Draws interactive UI components."""
    pygame.draw.rect(screen, BLUE, clear_button)
    pygame.draw.rect(screen, GREEN, train_button)
    pygame.draw.rect(screen, GRAY, input_box, 2)

    screen.blit(font.render("Clear", True, WHITE), (clear_button.x + 20, clear_button.y + 10))
    screen.blit(font.render("Train", True, WHITE), (train_button.x + 20, train_button.y + 10))
    screen.blit(font.render(text_input, True, WHITE), (input_box.x + 10, input_box.y + 10))

    pygame.draw.rect(screen, RED, (20, 20, 400, 30))
    screen.blit(font.render(prediction_result, True, WHITE), (30, 25))

running = True
while running:
    screen.fill(BLACK)
    draw_ui()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button.collidepoint(event.pos):
                points.clear()
            elif train_button.collidepoint(event.pos) and text_input.strip():
                train_doodle(points, text_input.strip())
                text_input = ""
                points.clear()
            elif input_box.collidepoint(event.pos):
                pass  
            else:
                drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                text_input = text_input[:-1]
            elif event.key == pygame.K_RETURN:
                train_doodle(points, text_input.strip())
                text_input = ""
                points.clear()
            else:
                text_input += event.unicode
        if event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.circle(screen, WHITE, event.pos, 5)
            points.append(event.pos)
            prediction_result = predict_doodle(points)  

    # Redraw all points to persist the doodle!
    for point in points:
        pygame.draw.circle(screen, WHITE, point, 5)

    pygame.display.update()

pygame.quit()
