import pygame
import json
import numpy as np

# Initialize pygame with resizable flag
pygame.init()
width, height = 500, 500
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
pygame.display.set_caption("trAIl")

# Colors
WHITE, BLACK, BLUE, GREEN, RED, GRAY, LIGHT_GRAY = (255, 255, 255), (0, 0, 0), (0, 128, 255), (0, 200, 0), (255, 60, 60), (150, 150, 150), (230, 230, 230)

# Font
font = pygame.font.Font(None, 32)

# Doodle storage
drawing = False
points = []
memory_file = "memory.json"
text_input = ""
prediction_result = ""

# Load memory
try:
    with open(memory_file, "r") as f:
        memory = json.load(f)
except FileNotFoundError:
    memory = {"doodles": []}
    with open(memory_file, "w") as f:
        json.dump(memory, f, indent=4)

def extract_features(points):
    """Extracts refined shape features."""
    if not points:
        return None
    x_vals, y_vals = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    width, height = max(x_vals) - min(x_vals), max(y_vals) - min(y_vals)
    density = len(points) / (width * height + 1)
    curve_factor = np.std(y_vals) / (np.std(x_vals) + 1)
    aspect_ratio = width / (height + 1)
    centroid_x, centroid_y = np.mean(x_vals), np.mean(y_vals)
    return {"width": int(width), "height": int(height), "density": float(density), "curve_factor": float(curve_factor),
            "aspect_ratio": float(aspect_ratio), "centroid_x": int(centroid_x), "centroid_y": int(centroid_y)}

def similarity_score(features1, features2):
    """Calculates cosine similarity."""
    f1, f2 = np.array(list(features1.values())), np.array(list(features2.values()))
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-5)

def predict_doodle(points):
    """Predicts a shape dynamically."""
    features = extract_features(points)
    if features is None:
        return "No doodle detected"
    best_match, best_score = None, -1  
    for prev_doodle in memory["doodles"]:
        prev_features = {key: prev_doodle.get(key, 0.0) for key in features.keys()}
        score = similarity_score(features, prev_features)
        if score > best_score:
            best_score, best_match = score, prev_doodle["label"]
    return f"Predicted: {best_match} ({best_score:.2f})" if best_match else "Unknown Shape - Train Me!"

def train_doodle(points, label):
    """Trains a doodle intelligently."""
    features = extract_features(points)
    if features and label:
        memory["doodles"].append({**features, "label": label})
        with open(memory_file, "w") as f:
            json.dump(memory, f, indent=4)

# Define UI elements globally
clear_button = pygame.Rect(50, height - 50, 120, 40)
train_button = pygame.Rect(180, height - 50, 120, 40)
input_box = pygame.Rect(width - 180, height - 50, 160, 40)

def draw_ui():
    """Draws UI components dynamically based on window size."""
    global width, height, clear_button, train_button, input_box  
    width, height = screen.get_size()  

    # Update UI positions dynamically
    clear_button = pygame.Rect(50, height - 50, 120, 40)
    train_button = pygame.Rect(180, height - 50, 120, 40)
    input_box = pygame.Rect(width - 180, height - 50, 160, 40)

    pygame.draw.rect(screen, BLUE if not clear_button.collidepoint(pygame.mouse.get_pos()) else LIGHT_GRAY, clear_button, border_radius=10)
    pygame.draw.rect(screen, GREEN if not train_button.collidepoint(pygame.mouse.get_pos()) else LIGHT_GRAY, train_button, border_radius=10)
    pygame.draw.rect(screen, GRAY, input_box, 2)

    screen.blit(font.render("Clear", True, WHITE), (clear_button.x + 20, clear_button.y + 10))
    screen.blit(font.render("Train", True, WHITE), (train_button.x + 20, train_button.y + 10))
    screen.blit(font.render(text_input, True, WHITE), (input_box.x + 10, input_box.y + 10))

    pygame.draw.rect(screen, RED, (20, 20, width - 40, 30))
    screen.blit(font.render(prediction_result, True, WHITE), (30, 25))

running = True
while running:
    screen.fill(BLACK)
    draw_ui()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button.collidepoint(event.pos):
                points.clear()
            elif train_button.collidepoint(event.pos) and text_input.strip():
                train_doodle(points, text_input.strip())
                text_input, points = "", []
            elif input_box.collidepoint(event.pos):
                pass  
            else:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                text_input = text_input[:-1]
            elif event.key == pygame.K_RETURN:
                train_doodle(points, text_input.strip())
                text_input, points = "", []
            else:
                text_input += event.unicode
        elif event.type == pygame.MOUSEMOTION and drawing:
            if points:
                pygame.draw.line(screen, WHITE, points[-1], event.pos, 3)
            points.append(event.pos)
            prediction_result = predict_doodle(points)  

    # Redraw all points smoothly
    for i in range(1, len(points)):
        pygame.draw.line(screen, WHITE, points[i - 1], points[i], 3)

    pygame.display.update()

pygame.quit()
