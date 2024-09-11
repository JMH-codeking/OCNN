import torch
from rich.progress import Progress
from matplotlib import pyplot as plt
import os
# def Dice_score(pred, target):
#   with torch.no_grad():
#     smooth = 0.01
    
#     iflat = pred.contiguous()
#     tflat = target.contiguous()
#     intersection = torch.sum(iflat * tflat, dim=[1,2,3])
#     iflat_sum = torch.sum(iflat, dim=[1,2,3])
#     tflat_sum = torch.sum(tflat, dim=[1,2,3])
#     return torch.mean((2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth))
def Dice_score(pred, target, smooth=1e-6):
    # 将预测和目标展平为一维向量
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 计算交集（intersection）和并集（sum of predictions and targets）
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    
    # 计算 Dice Score
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice


def train_model(model, dataloader, dataloader_test, criterion, optimizer,model_dir, num_epochs=50,restore_model=True, device = torch.device('cuda')):
  train_accs = []
  test_accs = []
  with Progress() as progress:
    model.train()
    best_acc = 0.0
    if restore_model:
      saved_models = sorted(os.listdir(model_dir), key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
      if len(saved_models) > 0:
        # Load the last saved model
        model_path = os.path.join(model_dir, saved_models[-1])
        model.load_state_dict(torch.load(model_path))
        print(f"Restored model from {model_path}")
 
        # Parse the epoch number from the model filename
        epoch_start = int(saved_models[-1].split('_')[2])  # Assumes format 'model_epoch_{epoch}_acc_{acc}.pth'
        print(f"Resuming training from epoch {epoch_start}")
    else:
      epoch_start=0
 
    epoch_task = progress.add_task(f'Training in progress ...', total = num_epochs)
    for epoch in range(epoch_start,num_epochs):
      running_loss = 0.0
      acc_tot = 0
      for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print (outputs.shape)
        # print (labels.shape)
        pred = torch.argmax(outputs, dim=1)
        acc = Dice_score(pred, labels)
        # print (acc)
        acc_tot += acc
        # print (outputs.shape)
        # print ('s')
        labels = labels.squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      progress.advance(epoch_task)
      avg_loss = running_loss / len(dataloader)
      avg_acc = acc_tot / len(dataloader)
      running_loss_test = 0.0
      acc_tot_test = 0
      for images, labels in dataloader_test:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          pred = torch.argmax(outputs, dim=1)
          acc = Dice_score(pred, labels)
          acc_tot_test += acc
          labels = labels.squeeze(1)
          loss = criterion(outputs, labels)
          running_loss_test += loss.item()
      avg_loss_test = running_loss_test / len(dataloader_test)
      avg_acc_test = acc_tot_test / len(dataloader_test)
    #   if avg_acc_test > 0.75:
    #     for images, labels in dataloader_test:
    #       images = images.to(device)
    #       labels = labels.to(device)
    #       outputs = model(images)
    #       acc = Dice_score(outputs, labels)
    #       print (acc)
    #       for cnt in range (20):
    #         print (cnt)
    #         cnt += 1
    #         plt.imshow(labels[cnt].cpu().detach().numpy().reshape(128, 128), cmap='gray')
    #         plt.savefig(f'./results/origin_{cnt}.png')
    #         plt.figure()
    #         plt.imshow(outputs[cnt].cpu().detach().numpy().reshape(128, 128), cmap='gray')
    #         plt.savefig(f'./results/output_{cnt}.png')
    #       quit()

      train_accs.append(avg_acc.cpu())
      test_accs.append(avg_acc_test.cpu())
      if epoch == 0:
        print (f'Epoch {epoch + 1} has  training Loss: {avg_loss:.4f}, acc: {avg_acc:.4f} and testing Loss: {avg_loss_test:.4f}, Acc: {avg_acc_test:.4f}')
      if (epoch + 1) % 10  == 0:
        print (f'Epoch {epoch + 1} has  training Loss: {avg_loss:.4f}, acc: {avg_acc:.4f} and testing Loss: {avg_loss_test:.4f}, Acc: {avg_acc_test:.4f}')
      # Save model if it's the best so far
      if avg_acc_test > best_acc:
        best_acc = avg_acc_test
        save_model(model, model_dir, f'model_epoch_{epoch+1:05}_acc_{avg_acc_test:.4f}.pth')
 
      if acc_tot/len(dataloader) > 0.99:
        return 0
  return train_accs, test_accs
 
 
def save_model(model, model_dir, model_name,max_keep=3):
  # Create the directory if it doesn't exist
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
 
  # Save the model
  model_path = os.path.join(model_dir, model_name)
  torch.save(model.state_dict(), model_path)
 
  # Get all saved models and sort them by creation time
  saved_models = sorted(os.listdir(model_dir), key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
 
  # If there are more than 3 models, delete the oldest ones
  if len(saved_models) > max_keep:
    # print('del:',saved_models[0])
    os.remove(os.path.join(model_dir, saved_models[0]))
  return 0