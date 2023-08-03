import torch
import torchvision
from torchvision import transforms
import numpy as np
from flask import Flask, render_template, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class PneumoniaModel():
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)

def load_file(path):
    return np.load(path).astype(np.float32)

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.49], [0.248]),
])

pytorch_model = PneumoniaModel().model
pt_file_path = "weights/weights_3.pth"
pytorch_model.load_state_dict(torch.load(pt_file_path))
pytorch_model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            try:
                image_array = load_file(filename)

                tensor_image = val_transforms(image_array)
                tensor_image = tensor_image.unsqueeze(0)

                with torch.no_grad():
                    prediction = pytorch_model(tensor_image)

                predicted_prob = torch.sigmoid(prediction).item()

                if predicted_prob < 0.4:
                    diagnosis = "No pneumonia"
                elif 0.4 <= predicted_prob <= 0.5:
                    diagnosis = "Maybe pneumonia"
                else:
                    diagnosis = "Pneumonia"

                result = f"Predicted Probability of Pneumonia: {predicted_prob:.2f}\nDiagnosis: {diagnosis}"
                return render_template('index.html', result=result)

            except FileNotFoundError:
                return "File not found, please try again.", 400

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
