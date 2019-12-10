import torch
import torch.nn.functional as F

from module.dice_loss import dice_coeff
from module.visualize_attention import visualize_Seg, visualize_Rec
import module.common_module as cm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import module.evaluation_voxel as evaluation
# import module.evaluation_lesion as evaluation
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import skimage.measure


def eval_net_dice(net, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None, epoch=None, slice=20, root_path='no_root_path'):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    running_loss = 0
    for i, b in enumerate(dataset):
        img = b['image'][:, 2:3].float()
        true_mask = b['mask'][:].float()

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img, phase, network_switch)[0]
        # mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
        # test_image = np.transpose((true_mask.cpu().detach().numpy()[0]).astype(float), [1, 2, 0])
        # result_image = np.transpose((mask_pred.cpu().detach().numpy()[0][0]).astype(float), [1, 2, 0])
        #
        # tot += evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))[0]

        loss = criterion[0](mask_pred.float(), true_mask.float())
        running_loss += loss.item() * img.size(0)

        if preview:
            if visualize_batch is not None:
                if i == int(visualize_batch):
                    outputs_vis = mask_pred.cpu().detach().numpy()
                    inputs_vis = img.cpu().detach().numpy()
                    labels_vis = true_mask.cpu().detach().numpy()
                    fig = visualize_Seg(inputs_vis[0][0], labels_vis[0], outputs_vis[0][0], figsize=(6, 6), epoch=epoch, slice=slice)

                    cm.mkdir(root_path + 'preview/val/Labeled')
                    plt.savefig(root_path + 'preview/val/Labeled/' + 'epoch_%s.jpg' % epoch)
                    # plt.show(block=False)
                    # plt.pause(1.0)
                    plt.close(fig)

    if i == 0:
        tot = tot
        running_loss = running_loss
    else:
        tot = tot / (i+1)
        running_loss = running_loss / (i+1)

    return tot, running_loss


def eval_net_mse(net, criterion, phase, network_switch, dataset, preview=True, gpu=False, visualize_batch=None, epoch=None, slice=20, root_path='no_root_path'):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    running_loss = 0
    for i, b in enumerate(dataset):
        img = b['image'][:, 2:3].float()
        true_mask = b['image'][:, 2:3].float()

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_predL, mask_pred = net(img, phase, network_switch)
        # mask_pred = (mask_pred > 0.5).float()
        mask_pred = mask_pred.float()

        outputsU_back = mask_pred[:, 0:1]
        outputsU_1 = mask_pred[:, 1:2]
        labels_back = (1.0 - mask_predL) * true_mask.float()
        labels_1 = mask_predL * true_mask.float()

        loss = criterion[1](outputsU_1.float(), labels_1.float())
        running_loss += loss.item() * img.size(0)

        if preview:
            if visualize_batch is not None:
                if i == int(visualize_batch):
                    outputsU_back_vis = outputsU_back.cpu().detach().numpy()
                    outputsU_1_vis = outputsU_1.cpu().detach().numpy()
                    inputs_vis = img.cpu().detach().numpy()
                    labels_back_vis = labels_back.cpu().detach().numpy()
                    labels_1_vis = labels_1.cpu().detach().numpy()
                    fig = visualize_Rec(inputs_vis[0][0], labels_back_vis[0, 0], labels_1_vis[0, 0],
                                        outputsU_back_vis[0, 0], outputsU_1_vis[0, 0], figsize=(6, 6), epoch=epoch)
                    cm.mkdir(root_path + 'preview/val/Unlabeled')
                    plt.savefig(root_path + 'preview/val/Unlabeled/' + 'epoch_%s.jpg' % epoch)
                    # plt.show(block=False)
                    # plt.pause(1.0)
                    plt.close(fig)

    if i == 0:
        running_loss = running_loss
    else:
        running_loss = running_loss / (i+1)

    return running_loss, running_loss


def test_net_dice(root_path, basic_path, model, network_switch, dataset, TSNE, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    with torch.no_grad():
        phase = 'trainLabeled'
        model.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))
        model.eval()
        DSC, AVD, Recall, F1 = 0, 0, 0, 0
        iter = 0
        i = 0

        tSNE = []
        tSNE_labels = []

        file = open(basic_path + '/test_results_lists.txt', 'a')

        for i, b in enumerate(tqdm(dataset)):
            DSC_1, AVD_1, Recall_1, F1_1 = 0, 0, 0, 0
            img = b['image'][:, 2:3].float()
            true_mask = b['mask'][0]

            # create prediction tensor
            prediction = torch.zeros(true_mask.size(), dtype=torch.float)

            imgShape = img.size()[-3:]
            resultShape = cm.BraTSshape

            imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
            resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

            overlapZ = 0.5
            overlapH = 0.5
            overlapW = 0.5

            interZ = int(resultShape[0] * (1.0 - overlapZ))
            interH = int(resultShape[1] * (1.0 - overlapH))
            interW = int(resultShape[2] * (1.0 - overlapW))

            iterZ = int(((imgShape[0]-resultShape[0]) / interZ)+1)
            iterH = int(((imgShape[1]-resultShape[1]) / interH)+1)
            iterW = int(((imgShape[2]-resultShape[2]) / interW)+1)

            freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
            freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
            freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

            startZ = int(freeZ/2)
            startH = int(freeH/2)
            startW = int(freeW/2)

            for z in range(0, iterZ):
                for h in range(0, iterH):
                    for w in range(0, iterW):
                        input = img[:, :, (startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                        label = true_mask[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))]

                        if gpu:
                            input = input.cuda()

                        if TSNE:
                            outputsL = model(input, phase, network_switch)[2][0].cpu()
                            # for i in range(mask_pred.shape[0]):
                            #     for j in range(mask_pred.shape[2]):
                            #         for k in range(mask_pred.shape[3]):
                            #     sample = mask_pred.detach().numpy()[i].flatten()
                            #     tSNE.append(sample)
                            mask_label = label.detach().numpy()

                            layers = 1

                            for i in range(1, layers):
                                mask_label = skimage.measure.block_reduce(mask_label, (2, 2, 2), np.mean)

                            for i in range(outputsL.shape[1]):
                                for j in range(outputsL.shape[2]):
                                    for k in range(outputsL.shape[3]):

                                        sample = outputsL.detach().numpy()[:, i, j, k].flatten()
                                        tSNE.append(sample)

                                        sample_label = mask_label[i, j, k]
                                        tSNE_labels.append(sample_label)

                        else:
                            outputsL, outputsU = model(input, phase=phase, network_switch=network_switch)
                            # outputsU_back = outputsU[:, 0]
                            # outputsU_fore = outputsU[:, 1]
                            # labels_back = (1.0 - outputsL) * input.float()
                            # labels_fore = outputsL * input.float()
                            #
                            # outputsL_vis = outputsL.cpu().detach().numpy()
                            # outputsU_back_vis = outputsU_back.cpu().detach().numpy()
                            # outputsU_fore_vis = outputsU_fore.cpu().detach().numpy()
                            # inputs_vis = input.cpu().detach().numpy()
                            # labels_vis = label.cpu().detach().numpy()
                            # labels_back_vis = labels_back.cpu().detach().numpy()
                            # labels_fore_vis = labels_fore.cpu().detach().numpy()
                            #
                            # # visualize training set at the end of each epoch
                            # cm.mkdir(root_path + 'preview/' + 'image_' + str(i))
                            #
                            # fig = visualize_Seg(inputs_vis[0][0], labels_vis, outputsL_vis[0][0],
                            #                     figsize=(6, 6), epoch=0)
                            # plt.savefig(root_path + 'preview/' + 'image_' + str(i) + '/Seg_{}{}{}.jpg'.format(z,h,w))
                            # plt.close(fig)
                            #
                            #
                            # fig = visualize_Rec(inputs_vis[0][0], labels_back_vis[0][0],
                            #                     labels_fore_vis[0][0],
                            #                     outputsU_back_vis[0], outputsU_fore_vis[0],
                            #                     figsize=(6, 6),
                            #                     epoch=0)
                            # plt.savefig(root_path + 'preview/' + 'image_' + str(i) + '/Rec_{}{}{}.jpg'.format(z,h,w))
                            # plt.close(fig)


                        if not TSNE:
                            prediction[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                    (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                    (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))] += outputsL[0][0].cpu()
                            imgMatrix[(startZ + (interZ*z)):(startZ + (interZ*(z) + resultShape[0])),
                                    (startH + (interH*h)):(startH + (interH*(h) + resultShape[1])),
                                    (startW + (interW*w)):(startW + (interW*(w) + resultShape[2]))] += resultMatrix

            if not TSNE:
                imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
                result = np.divide(prediction.cpu().detach().numpy(), imgMatrix)
                test_image = np.transpose((np.where(true_mask.cpu().detach().numpy() > 0.5, 1, 0)).astype(int), [1, 2, 0])
                result_image = np.transpose((np.where(result > 0.5, 1, 0)).astype(int), [1, 2, 0])

                dsc, avd, recall, f1 = evaluation.do(sitk.GetImageFromArray(test_image), sitk.GetImageFromArray(result_image))
                DSC += dsc
                AVD += avd
                Recall += recall
                F1 += f1

                DSC_1 += dsc
                AVD_1 += avd
                Recall_1 += recall
                F1_1 += f1

                history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(DSC_1, AVD_1, Recall_1, F1_1))
                file.write(history)

        file.close()

        total = (iter + 1) * (i + 1)

        if TSNE:
            TSNE_array = np.ndarray([len(tSNE), sample.shape[0]])
            TSNE_label_array = np.ndarray([len(tSNE_labels)])
            for i in range(len(tSNE)):
                TSNE_array[i] = tSNE[i]
                TSNE_label_array[i] = tSNE_labels[i]

            np.save(root_path + '/tSNE.npy', TSNE_array)
            np.save(root_path + '/tSNE_labels.npy', TSNE_label_array)

    return DSC/total, AVD/total, Recall/total, F1/total
