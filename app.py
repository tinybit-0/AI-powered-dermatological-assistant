import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import threading
import os

class APPv2:
    def __init__(self, root):
        self.root = root
        self.root.title("Analyzer")
        self.root.geometry("600x800")
        self.root.resizable(False, False)

        self.main_frame = tk.Frame(root, bg="#f0f0f0", padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(self.main_frame, text="Analyzer", font=("Inter", 20, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=(0, 10))

        self.desc_label = tk.Label(self.main_frame, text="Upload image", font=("Inter", 12), bg="#f0f0f0")
        self.desc_label.pack(pady=(0, 20))

        self.image_canvas = tk.Canvas(self.main_frame, width=400, height=300, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.image_canvas.pack(pady=10)
        self.image_canvas.create_text(200, 150, text="Image", font=("Inter", 12), fill="#aaaaaa")

        # Buttons
        self.button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.button_frame.pack(pady=20)
        
        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image, font=("Inter", 12), bg="#e0e0e0", fg="#333333", relief="raised")
        self.upload_button.pack(side=tk.LEFT, padx=10)

        self.analyze_button = tk.Button(self.button_frame, text="Analyze", command=self.start_analysis, font=("Inter", 12), bg="#1e90ff", fg="white", relief="raised", state=tk.DISABLED)
        self.analyze_button.pack(side=tk.BOTTOM, padx=10)

        self.loading_label = tk.Label(self.main_frame, text="", font=("Inter", 12), fg="#1e90ff", bg="#f0f0f0")
        self.loading_label.pack(pady=10)

        self.results_label = tk.Label(self.main_frame, text="Results", font=("Inter", 16, "bold"), bg="#f0f0f0")
        self.results_label.pack(pady=(10, 5))
        
        self.results_text = tk.Text(self.main_frame, wrap=tk.WORD, height=15, bg="#ffffff", bd=1, relief="solid", padx=10, pady=10, font=("Inter", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.insert(tk.END, "Results")

        self.pil_image = None
        self.image_filename = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = self.load_model()
    
    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'skincancerclassifier.pth')
        #print(f"Attempting to load model from: {model_path}")

        try:
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(self.class_names))

            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
           # print("Model loaded successfully.")
            return model
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load the model file 'skin_lesion_classifier.pth'. Please ensure it is in the same directory as the script. Error: {e}")
            return None

    def upload_image(self):
        self.image_filename = filedialog.askopenfilename(
            initialdir="./",
            title="Select an image",
            filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("all files", "*.*"))
        )
        if self.image_filename:
            try:
                self.pil_image = Image.open(self.image_filename).convert('RGB')
                
                width, height = self.pil_image.size
                canvas_width = self.image_canvas.winfo_width()
                canvas_height = self.image_canvas.winfo_height()
                aspect_ratio = width / height
                
                if width > height:
                    new_width = canvas_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = canvas_height
                    new_width = int(new_height * aspect_ratio)
                
                resized_image = self.pil_image.resize((new_width, new_height), Image.LANCZOS)
                self.tk_image = ImageTk.PhotoImage(resized_image)
                
                self.image_canvas.delete("all")
                self.image_canvas.create_image(canvas_width/2, canvas_height/2, image=self.tk_image, anchor=tk.CENTER)
                
                self.analyze_button.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded. CLick Analyze")

            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")
                self.pil_image = None
                self.analyze_button.config(state=tk.DISABLED)

    def start_analysis(self):
        if self.pil_image:
            self.analyze_button.config(state=tk.DISABLED)
            self.upload_button.config(state=tk.DISABLED)
            self.loading_label.config(text="Analyzing image")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analyzing image")
            
            analysis_thread = threading.Thread(target=self.analyze_image, daemon=True)
            analysis_thread.start()

    def analyze_image(self):
        if self.model is None:
            self.root.after(0, self.update_results, "Model not loaded. Please restart the application.")
            return

        try:
            image_tensor = self.preprocess(self.pil_image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                confidence, predicted_class_idx = torch.max(probabilities, 0)
            
            predicted_class_name = self.class_names[predicted_class_idx.item()]
            confidence_score = confidence.item()
            
            result_text = (
                f"Analysis Complete!\n\n"
                f"Predicted Class: {predicted_class_name}\n"
                f"Confidence: {confidence_score:.4f}\n\n"
            )
            
            self.root.after(0, self.update_results, result_text)
            
        except Exception as e:
            error_message = f"An error occurred during model inference: {e}"
            self.root.after(0, self.update_results, error_message)

    def update_results(self, text):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.loading_label.config(text="")
        self.analyze_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = APPv2(root)
    root.mainloop()
