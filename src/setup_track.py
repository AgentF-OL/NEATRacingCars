import pygame
import json
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_vars import *

pygame.init()
pygame.font.init()

FONT = pygame.font.SysFont("consolas", 24)
SMALL_FONT = pygame.font.SysFont("consolas", 18)

# ── helpers ──
def _rotate_triangle(center, angle_deg, size=14):
    """Return 3 points of an isoceles triangle pointing 'up' (0°) then rotated."""
    rad = math.radians(angle_deg)
    base = [(0, -size), (-size * 0.7, size * 0.6), (size * 0.7, size * 0.6)]
    cx, cy = center
    pts = []
    for x, y in base:
        xr = x * math.cos(rad) - y * math.sin(rad)
        yr = x * math.sin(rad) + y * math.cos(rad)
        pts.append((cx + xr, cy + yr))
    return pts


class TrackSetup:
    PHASES = ["finish", "waypoints", "car"]
    MOVE_SPEED = 3
    FINE_SPEED = 1
    ROT_SPEED = 2.5

    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Track Setup")
        self.clock = pygame.time.Clock()

        self.phase_idx = 0
        self.mode = "position"
        self.show_hud = True          # ← NEW: toggle for instructions panel

        self.finish_pos = [FINISH_POSITION[0] + FINISH.get_width() // 2,
                           FINISH_POSITION[1] + FINISH.get_height() // 2]
        self.finish_angle = 0.0
        self.waypoints = []
        self.car_pos = None
        self.car_angle = 0.0

        self.config_path = "track_config.json"
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    cfg = json.load(f)
                self.finish_pos = list(cfg.get("finish_position", self.finish_pos))
                self.finish_angle = cfg.get("finish_angle", 0.0)
                self.waypoints = [tuple(p) for p in cfg.get("waypoints", [])]
                if "car_start" in cfg:
                    self.car_pos = list(cfg["car_start"])
                self.car_angle = cfg.get("car_angle", 0.0)
            except Exception as e:
                print(f"[setup] Could not load previous config: {e}")

    @property
    def phase(self):
        return self.PHASES[self.phase_idx] if self.phase_idx < len(self.PHASES) else "done"

    def _finish_topleft(self):
        rot_finish = pygame.transform.rotate(FINISH, self.finish_angle)
        w, h = rot_finish.get_width(), rot_finish.get_height()
        return (self.finish_pos[0] - w // 2, self.finish_pos[1] - h // 2)

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)

            # ── Continuous movement ──
            keys = pygame.key.get_pressed()
            shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
            speed = self.FINE_SPEED if shift else self.MOVE_SPEED

            if self.phase == "finish":
                if self.mode == "position":
                    if keys[pygame.K_UP]:
                        self.finish_pos[1] -= speed
                    if keys[pygame.K_DOWN]:
                        self.finish_pos[1] += speed
                    if keys[pygame.K_LEFT]:
                        self.finish_pos[0] -= speed
                    if keys[pygame.K_RIGHT]:
                        self.finish_pos[0] += speed
                else:
                    if keys[pygame.K_LEFT]:
                        self.finish_angle -= self.ROT_SPEED
                    if keys[pygame.K_RIGHT]:
                        self.finish_angle += self.ROT_SPEED

            elif self.phase == "car":
                if self.mode == "position" and self.car_pos:
                    if keys[pygame.K_UP]:
                        self.car_pos[1] -= speed
                    if keys[pygame.K_DOWN]:
                        self.car_pos[1] += speed
                    if keys[pygame.K_LEFT]:
                        self.car_pos[0] -= speed
                    if keys[pygame.K_RIGHT]:
                        self.car_pos[0] += speed
                elif self.mode == "rotation" and self.car_pos:
                    if keys[pygame.K_LEFT]:
                        self.car_angle -= self.ROT_SPEED
                    if keys[pygame.K_RIGHT]:
                        self.car_angle += self.ROT_SPEED

            # ── Events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if self.phase == "waypoints" and len(self.waypoints) == 0:
                            continue
                        self.phase_idx += 1
                        if self.phase == "done":
                            self.save()
                            return True

                    elif event.key == pygame.K_BACKSPACE:
                        if self.phase == "waypoints" and self.waypoints:
                            self.waypoints.pop()
                        elif self.phase == "car":
                            self.car_pos = None

                    elif event.key in (pygame.K_h, pygame.K_TAB):   # ← NEW: toggle HUD
                        self.show_hud = not self.show_hud

                    elif event.key == pygame.K_a:
                        self.mode = "rotation" if self.mode == "position" else "position"

                    elif event.key == pygame.K_r:
                        self.phase_idx = 0
                        self.mode = "position"
                        self.finish_pos = [FINISH_POSITION[0] + FINISH.get_width() // 2,
                                           FINISH_POSITION[1] + FINISH.get_height() // 2]
                        self.finish_angle = 0.0
                        self.waypoints = []
                        self.car_pos = None
                        self.car_angle = 0.0

                    elif event.key == pygame.K_ESCAPE:
                        return False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if event.button == 1:
                        if self.phase == "waypoints":
                            self.waypoints.append((mx, my))
                        elif self.phase == "car":
                            self.car_pos = [mx, my]
                    elif event.button == 3:
                        if self.phase == "waypoints":
                            to_remove = None
                            best_dist = 20
                            for i, (wx, wy) in enumerate(self.waypoints):
                                d = math.hypot(mx - wx, my - wy)
                                if d < best_dist:
                                    best_dist = d
                                    to_remove = i
                            if to_remove is not None:
                                self.waypoints.pop(to_remove)

            self.draw()

        return False

    def draw(self):
        # ── Clear background ──
        self.win.blit(TRACK, (0, 0))

        # ── Waypoints ──
        if len(self.waypoints) > 1:
            pygame.draw.lines(self.win, (0, 200, 0), False, self.waypoints, 2)
        for i, (x, y) in enumerate(self.waypoints):
            color = (0, 255, 0) if self.phase == "waypoints" else (0, 150, 0)
            pygame.draw.circle(self.win, color, (x, y), 6)
            txt = SMALL_FONT.render(str(i), True, (255, 255, 255))
            self.win.blit(txt, (x + 8, y - 8))

        # ── Finish line ──
        rot_finish = pygame.transform.rotate(FINISH, self.finish_angle)
        fx, fy = self._finish_topleft()
        self.win.blit(rot_finish, (fx, fy))
        if self.phase == "finish":
            w, h = rot_finish.get_width(), rot_finish.get_height()
            pygame.draw.rect(self.win, (255, 0, 0), (fx, fy, w, h), 2)
            cx, cy = self.finish_pos
            pygame.draw.line(self.win, (255, 0, 0), (cx - 10, cy), (cx + 10, cy), 2)
            pygame.draw.line(self.win, (255, 0, 0), (cx, cy - 10), (cx, cy + 10), 2)

        # ── Car start (triangle) ──
        car = self.car_pos if self.car_pos else [180, 200]
        tri = _rotate_triangle(car, self.car_angle, size=16)
        pygame.draw.polygon(self.win, (0, 150, 255), tri)
        pygame.draw.polygon(self.win, (255, 255, 255), tri, 2)
        if self.phase == "car" and self.car_pos:
            txt = SMALL_FONT.render("START", True, (0, 150, 255))
            self.win.blit(txt, (car[0] + 18, car[1] - 10))

        # ── HUD Panel ──
        if self.show_hud:
            panel_x = WIDTH - 300
            panel_w = 290
            pygame.draw.rect(self.win, (20, 20, 30), (panel_x - 10, 40, panel_w, 340))
            pygame.draw.rect(self.win, (80, 80, 100), (panel_x - 10, 40, panel_w, 340), 2)

            lines = [
                f"PHASE : {self.phase.upper()}",
                f"MODE  : {self.mode.upper()} (press A)",
                "",
                f"Finish: ({self.finish_pos[0]:.0f},{self.finish_pos[1]:.0f})",
                f"        angle={self.finish_angle:.1f}°",
                f"Waypoints: {len(self.waypoints)}",
                f"Car start: {tuple(map(int, self.car_pos)) if self.car_pos else 'default'}",
                f"          angle={self.car_angle:.1f}°",
                "",
                "CONTROLS",
                "  ARROWS : move / rotate",
                "  SHIFT  : fine speed",
                "  A      : toggle pos/rot",
                "  H      : hide/show HUD",
                "  ENTER  : next phase",
                "  BACKSP : undo / clear",
                "  R      : reset all",
                "  ESC    : cancel",
            ]
            if self.phase == "waypoints":
                lines.append("  CLICK  : add waypoint")
                lines.append("  R-CLICK: remove waypoint")

            for i, line in enumerate(lines):
                color = (220, 220, 220) if not line.startswith("  ") else (180, 180, 180)
                if line.startswith("CONTROLS"):
                    color = (255, 220, 100)
                surf = SMALL_FONT.render(line, True, color)
                self.win.blit(surf, (panel_x, 50 + i * 20))
        else:
            # Minimal hint when HUD is hidden
            hint = SMALL_FONT.render("Press H or TAB to show controls", True, (200, 200, 200))
            self.win.blit(hint, (WIDTH - hint.get_width() - 10, 10))

        # ── Bottom status bar ──
        status = f"Adjusting: {self.phase.upper()} | Mode: {self.mode.upper()}"
        if self.phase == "finish":
            status += "  |  Center crosshair = finish center"
        elif self.phase == "car":
            status += "  |  Blue triangle = car start"
        bar = pygame.Surface((WIDTH, 32))
        bar.fill((30, 30, 40))
        self.win.blit(bar, (0, HEIGHT - 32))
        txt = FONT.render(status, True, (200, 200, 200))
        self.win.blit(txt, (10, HEIGHT - 28))

        pygame.display.update()

    def save(self):
        data = {
            "finish_position": tuple(self.finish_pos),
            "finish_angle": float(self.finish_angle),
            "waypoints": self.waypoints,
            "car_start": tuple(self.car_pos) if self.car_pos else (180, 200),
            "car_angle": float(self.car_angle),
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[setup] Saved to {self.config_path}")
        print(f"  Finish center : {data['finish_position']}")
        print(f"  Finish angle  : {data['finish_angle']:.1f}°")
        print(f"  Waypoints     : {len(data['waypoints'])}")
        print(f"  Car start     : {data['car_start']}")
        print(f"  Car angle     : {data['car_angle']:.1f}°")


def main():
    setup = TrackSetup()
    success = setup.run()
    pygame.quit()
    if success:
        print("\nReady to run:")
        print("  python src/train_unified.py --strategy waypoints --generations 100")
        print("  python src/demo_winner.py waypoints --final")
    else:
        print("\nSetup cancelled.")


if __name__ == "__main__":
    main()