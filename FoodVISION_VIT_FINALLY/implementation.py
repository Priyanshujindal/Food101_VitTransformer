import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
class VitTransformer(nn.Module):
    def __init__(self,model_path:str,device:str='cpu'):
        super().__init__()
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=None
        self.classes=[]
        self.transform=None
        self._load_model(model_path)
        self._set_transform()
    def _load_model(self,model_path:str):
         self.model=models.vit_b_16(weights=None)
         checkpoints=torch.load(model_path,map_location=self.device)
         self.model.heads.head=nn.Linear(in_features=self.model.heads.head.in_features,out_features=101)
         self.model.load_state_dict(checkpoints['model_dict'])
         self.model.to(self.device)
         self.classes=checkpoints['classes']
    def _set_transform(self):
        self.transform=transforms.Compose([
          transforms.Resize((224,224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def predict(self,image):
        try:
            self.model.eval()
            img=image.convert('RGB')
            img=self.transform(img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits=self.model(img)
                probabilities=torch.softmax(logits,dim=1)
                prediction=logits.argmax(dim=1)
                confidence=probabilities[0][prediction.item()].item()
                print(f"Prediction: {self.classes[prediction.item()]}, Confidence: {confidence}")

                return  self.classes[prediction.item()],confidence
        except Exception as e:
            print(f"Error in prediction: {e} classes length: {len(self.classes)}")
            return "No prediction",0
def predict_image(image,model_path:str='model.pth',device:str='cpu'):
    model=VitTransformer(model_path,device)
    return model.predict(image)

if __name__=='__main__':
    print("Vit Transformer Model")




