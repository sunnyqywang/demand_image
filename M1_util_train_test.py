import sys
sys.path.append("models/")
import importlib
import numpy as np
import time
import torch

def load_model(script, model, config):
    module = importlib.import_module(script)
    Network = getattr(module, model)
    return Network(config)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count
        
def train(epoch, model, optimizer, criterion, train_loader, model_regularizer, run_config,
          writer, device, logger=None):

    model.train()
    demo_cs, demo_np = model_regularizer

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    
    for step, (image_list, data) in enumerate(train_loader):
        
        if run_config['tensorboard'] and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        census_index = [demo_cs.index(i[i.rfind('/')+1:i.rfind('_')]) for i in image_list]
        census_data = demo_np[census_index]
        
        census_data = torch.tensor(census_data).to(device)
        data = data.to(device)
        
        optimizer.zero_grad()

        out_image, out_demo = model(data)

        loss = criterion(out_image, out_demo, data, census_data, factor=run_config['weight'], factorr=run_config['weightt'])
        loss.backward()

        optimizer.step()
        
        num = data.size(0)

        loss_ = loss.item()
        loss_meter.update(loss_, num)

#         _, preds = torch.max(outputs, dim=1)
#         correct_ = preds.eq(targets).sum().item()
#         accuracy = correct_ / num
#         accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, epoch)
#             writer.add_scalar('Train/RunningAccuracy', accuracy, epoch)

#     logger.info('Epoch {} Step {}/{} '
#                 'Loss {:.4f} ({:.4f}) '
#                 'Accuracy {:.4f} ({:.4f})'.format(
#                     epoch,
#                     step,
#                     len(train_loader),
#                     loss_meter.val,
#                     loss_meter.avg,
#                     accuracy_meter.val,
#                     accuracy_meter.avg,
#                 ))
  
    if logger is not None:
        logger.info('Epoch {} Step {}/{} '
                'Train Loss {:.8f}'.format(
                    epoch,
                    step,
                    len(train_loader),
                    loss_meter.avg
                ))

    if run_config['tensorboard']:
        elapsed = time.time() - start
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
#         writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)
    
    return loss_meter.avg

def test(epoch, model, criterion, test_loader, model_regularizer, run_config, writer, device, logger, return_output=False):

    model.eval()
    demo_cs, demo_np = model_regularizer
    return_images = []
    return_demo = []

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    for step, (image_list, data) in enumerate(test_loader):
        if run_config['tensorboard'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)

        data = data.to(device)
        census_index = [demo_cs.index(i[i.rfind('/')+1:i.rfind('_')]) for i in image_list]
        census_data = demo_np[census_index]
        
        census_data = torch.tensor(census_data).to(device)

        with torch.no_grad():            
            out_image, out_demo = model(data)
#         print(torch.max(out_image))
#         print(torch.min(out_image))
#         print(torch.max(data))
#         print(torch.min(data))
        
        loss = criterion(out_image, out_demo, data, census_data, factor=run_config['weight'], factorr=run_config['weightt'])
        loss_ = loss.item()
#         print(loss_)
        num = data.size(0)
        loss_meter.update(loss_, num)
#         print(loss_meter.avg)
        if return_output:
            return_images.append(out_image.detach().cpu().numpy())
            return_demo.append(out_demo.detach().cpu().numpy())

#     logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
#         epoch, loss_meter.avg, accuracy))

    if logger is not None:
        logger.info('Epoch {} Test Loss {:.8f}'.format(epoch, loss_meter.avg))
    
        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))
    else:
        print('Epoch {} Test Loss {:.8f}'.format(epoch, loss_meter.avg))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
#         writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    if return_output:
        return_images = np.vstack(return_images)
        return_demo = np.vstack(return_demo)
        return return_images, return_demo
    else:
        return loss_meter.avg
    