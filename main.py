import torch
import torch.optim as optim
import AutoEncoder
import MaskRcnn
import tools

device = torch.device('cuda')

model = AutoEncoder.AutoEncoder()

optimizer = optim.SGD(model.parameters(),lr = 1e-3)
criterion = torch.nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=5)

print("torch version:", torch.__version__)
print("cuda 사용 가능여부:", torch.cuda.is_available())  # GPU 사용 가능 여부
if torch.cuda.is_available():
    print("cuda 버전:", torch.version.cuda)  # CUDA 버전
    print("gpu 이름:", torch.cuda.get_device_name(0))  # GPU 이름
    device = torch.device("cuda")
else:
    print("plz use cpu")
    device = torch.device("cpu")

trainloader,testloader = tools.datamaker()

tools.model_train(model,device,criterion,optimizer,scheduler,trainloader,testloader)