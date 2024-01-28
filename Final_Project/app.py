import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import pygame
import cv2
import numpy

class DrowsyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("700x650")
        self.title("Drowsiness Detection")
        self.create_widgets()

    def create_widgets(self):
        # create video Frame
        vid_frame = tk.Frame(self, height=480, width=600)
        vid_frame.pack()
        self.vid_label = tk.Label(vid_frame)
        self.vid_label.pack()

        # create Label for showing the YOLOv8 process infromation
        self.info_label_yolo = tk.Label(self, text="", font=("Helvetica", 12))
        self.info_label_yolo.pack()

        # load YOLO model
        self.threshold = 0.5
        self.model = YOLO("weights/Hoa_kaggle_best_100ep.pt")

        # start video thread
        self.video_thread = threading.Thread(target=self.update_video)
        self.video_thread.daemon = True
        self.video_thread.start()

        # initialize sound
        pygame.init()
        pygame.mixer.init()
        self.counter = 0
        self.sound_file = "warning.wav"
        self.sound = pygame.mixer.Sound(self.sound_file)

    def play_warning_sound(self):
        self.sound.play()

    def update_video(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, original_frame = cap.read()
            original_frame = cv2.flip(original_frame, 1)

            # YOLO prediction
            results = self.model.predict(original_frame, verbose=False, max_det = 1, conf = self.threshold, line_width=2) 

            if ret:
                # warning
                if numpy.any(results[0].boxes.cpu().numpy().cls == 0):
                    if self.counter < 20:
                        self.counter += 1
                    if not pygame.mixer.get_busy() and self.counter >= 20:
                        self.play_warning_sound()
                else:
                    self.counter = 0
                    pygame.mixer.stop()

                # display output predicted frame
                display_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(Image.fromarray(display_frame))
                self.vid_label.config(image=photo)
                self.vid_label.image = photo

                # display YOLOv8 process infromation
                if results is not None and len(results) > 0:
                    speed_info = results[0].speed
                    info_text = "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms postprocess per image at shape {}".format(
                        speed_info["preprocess"],
                        speed_info["inference"],
                        speed_info["postprocess"],
                        results[0].orig_shape
                    )
                    self.info_label_yolo.config(text=info_text)

            self.update()

        cap.release()


if __name__ == "__main__":
    app = DrowsyApp()
    app.mainloop()
