import numpy as np
import sys
import os

np.set_printoptions(threshold=np.inf)
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import cv2
import warnings

warnings.filterwarnings("ignore")


'''
label_1 = [0, 0, 0]

label_2 = [128, 0, 0]
label_3 = [0,128,0]
label_4 = [128,128,0]
label_5 = [0,0,128]
label_6 = [128,0,128]
label_7 = [0,128,128]
label_8 = [128,128,128]
label_9 = [64,0,0]
label_10 = [192,0,0]
label_11 = [64, 128, 0]

label_12 = [192,128,0]
label_13 = [64,0,128]
label_14 = [192,0,128]
label_15 = [64,128,128]
label_16 = [192, 128, 128]
label_17 = [0, 64, 0]
label_18 = [128, 64, 0]
label_19 = [0, 192, 0]
label_20 = [128, 192, 0]
'''

label_list = [[0, 0, 0], [0, 0, 128], [0, 128, 192], [128,128,128]]


SIZE_I = 256
NUM_I = 4

def imgToMatrix(PngPath) :
    # 将单张图片转换成矩阵(880*880*3)，同时返回w，h
    # 将单张图片转换成矩阵(256*256*3)，同时返回w，h
    # print(PngPath)
    img_matrix = cv2.imread(PngPath)
    #b, g, r = cv2.split(img_matrix)
    #img_rgb = cv2.merge([r, g, b])
    width = img_matrix.shape[0]  # 预测值长度
    height = img_matrix.shape[1]  # 预测值宽度
    # grayMatrix=np.zeros((width, height))
    # for i in range(0,width):
    #     for j in range(0,height):
    #         grayMatrix[i][j]=img_matrix[i][j][0]
    # print(grayMatrix)
    return img_matrix, width, height


def getOneVoxelFPTPTNFN(prediction, groundtrue, class_num=NUM_I, width=SIZE_I, height=SIZE_I) :
    # 输入的是880*880*3（rgb模式）
    # 输入的是256*256*3（rgb模式）
    # 获得单张切片的每个类的FP、TP、TN、FN，输入是单张切片的矩阵，图片固定则FPTPFNTN固定
    tp_allClass = np.zeros((class_num), dtype=np.float32)
    fp_allClass = np.zeros((class_num), dtype=np.float32)
    tn_allClass = np.zeros((class_num), dtype=np.float32)
    fn_allClass = np.zeros((class_num), dtype=np.float32)
    classN = 0
    for c in label_list :
        cc=np.array(c)
        # print('ftp')
        # print(c)
        # c-->[r,g,b]
        # 统计每个分类的TPFPTNFN
        flatten_gt = np.reshape(groundtrue, (width * height, 3))
        flatten_pr = np.reshape(prediction, (width * height, 3))
        if (not ((cc == flatten_gt).all(1).any()) and not ((cc == flatten_pr).all(1).any())) :
            # if c not in groundtrue and c not in prediction:
            # 跳过没出现在本tu中的分类
            classN+=1
            #print("nono",c)
            continue
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        width = groundtrue.shape[0]
        height = groundtrue.shape[1]
        for i in range(0, width) :  # 遍历所有长度的点
            for j in range(0, height) :  # 遍历所有宽度的点
                pix_pre = prediction[i][j]  # 提取预测图点的色彩值
                pix_tru = groundtrue[i][j]  # 提取真实图点的色彩值
                # 大小[3]
                if (pix_tru == cc).all() and (pix_pre == cc).all():  # 真实阳性 且 预测阳性
                    TP += 1
                elif (pix_tru != cc).any() and (pix_pre != cc).any():  # 真实阴性 且 预测阴性
                    TN += 1
                elif (pix_tru != cc).any() and (pix_pre == cc).all():  # 真实阴性 且 预测阳性
                    FP += 1
                elif (pix_tru == cc).all() and (pix_pre != cc).any():  # 真实阳性 且 预测阴性
                    FN += 1
        tp_allClass[classN] = TP
        fp_allClass[classN] = FP
        tn_allClass[classN] = TN
        fn_allClass[classN] = FN
        classN += 1
    #print(classN)
    return tp_allClass, fp_allClass, tn_allClass, fn_allClass


def oneVoxeLevelDice(prediction, groundtrue, tp_allClass, fp_allClass, tn_allClass, fn_allClass, class_num=NUM_I,
                     width=SIZE_I, height=SIZE_I) :
    # 计算单张切片的dice
    # 输入是预测以及GT的矩阵(880*880*3)，本张切片所有分类的的FNTNFPTP
    empty_value = -1.0
    dice_allClass = empty_value * np.ones((class_num), dtype=np.float32)
    eps = 1e-10
    i = 0
    for c in label_list :
        # print('dice')
        # print(c)
        # 计算每个分类的dice
        # flatten_gt = np.reshape(groundtrue,(width*height,3))
        # flatten_pr = np.reshape(prediction, (width * height, 3))
        # t=[128, 192, 0]
        # print((t!= flatten_gt).all(1).any())
        # print(flatten.shape)
        # print(c)

        flatten_gt = np.reshape(groundtrue, (width * height, 3))
        flatten_pr = np.reshape(prediction, (width * height, 3))
        # print(not(c == flatten_gt).all(1).any())
        if (not ((c == flatten_gt).all(1).any()) and not ((c == flatten_pr).all(1).any())) :
            # 跳过没出现在本例子中的分类
            i = i + 1
            continue
        tp = tp_allClass[i]
        fp = fp_allClass[i]
        tn = tn_allClass[i]
        fn = fn_allClass[i]
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        dice_allClass[i] = dice
        i = i + 1
        # print('tp_allClass:',tp_allClass)
        # print('tp', tp, 'fp', fp, 'tn', tn, 'fn', fn)
    # print('dice_allClass',dice_allClass)
    # print('1DICE')
    dscs = np.where(dice_allClass == -1.0, np.nan, dice_allClass)
    voxel_level_dice = np.nanmean(dscs[1 :])
    voxel_level_dice_std = np.nanstd(dscs[1 :], ddof=1)
    # print('voxel_level_dice_std',voxel_level_dice_std)
    return voxel_level_dice, voxel_level_dice_std


def oneVoxeLevelAccuracy(prediction, groundtrue, tp_allClass, fp_allClass, tn_allClass, fn_allClass, class_num=NUM_I,
                         width=SIZE_I, height=SIZE_I) :
    # 计算单张切片的Accuracy
    # 输入是预测以及GT的矩阵(880*880)，本张切片所有分类的的FNTNFPTP
    eps = 1e-10
    empty_value = -1.0
    acc_allClass = empty_value * np.ones((class_num), dtype=np.float32)
    i = 0
    for c in label_list :
        # 计算每个分类的Accuracy
        flatten_gt = np.reshape(groundtrue, (width * height, 3))
        flatten_pr = np.reshape(prediction, (width * height, 3))
        if (not ((c == flatten_gt).all(1).any()) and not ((c == flatten_pr).all(1).any())) :
            # 跳过没出现在本例子中的分类
            i = i + 1
            continue
        tp = tp_allClass[i]
        fp = fp_allClass[i]
        tn = tn_allClass[i]
        fn = fn_allClass[i]
        accracy = (tp + tn) / (tp + fp + tn + fn + eps)
        # print('acc iiis',accracy)
        acc_allClass[i] = accracy
        i = i + 1
        # print('tp',tp,'fp',fp,'tn',tn,'fn',fn)
    # print('acc_allClass', acc_allClass)
    accs = np.where(acc_allClass == -1.0, np.nan, acc_allClass)
    # print('accs[1:]',accs[1:])
    voxel_level_acc = np.nanmean(accs[1 :])
    voxel_level_acc_std = np.nanstd(accs[1 :], ddof=1)
    # print('voxel_level_acc_std',voxel_level_acc_std)
    # print('voxel_level_acc', voxel_level_acc)
    return voxel_level_acc, voxel_level_acc_std


def oneVoxeLevelPrecision(prediction, groundtrue, tp_allClass, fp_allClass, tn_allClass, fn_allClass, class_num=NUM_I,
                          width=SIZE_I, height=SIZE_I) :
    # 计算单张切片的Precision
    # 输入是预测以及GT的矩阵(880*880)，本张切片所有分类的的FNTNFPTP
    empty_value = -1.0
    precision_allClass = empty_value * np.ones((class_num), dtype=np.float32)
    eps = 1e-10
    i = 0
    for c in label_list :
        # 计算每个分类的Precision
        flatten_gt = np.reshape(groundtrue, (width * height, 3))
        flatten_pr = np.reshape(prediction, (width * height, 3))
        if (not ((c == flatten_gt).all(1).any()) and not ((c == flatten_pr).all(1).any())) :
            # 跳过没出现在本例子中的分类
            i = i + 1
            continue
        tp = tp_allClass[i]
        fp = fp_allClass[i]
        tn = tn_allClass[i]
        fn = fn_allClass[i]
        precision = tp / (tp + fp + eps)
        precision_allClass[i] = precision
        i = i + 1
    precisions = np.where(precision_allClass == -1.0, np.nan, precision_allClass)
    voxel_level_precision = np.nanmean(precisions[1 :])
    voxel_level_precision_std = np.nanstd(precisions[1 :], ddof=1)
    # print('voxel_level_precision_std',voxel_level_precision_std)
    return voxel_level_precision, voxel_level_precision_std


def oneVoxeLevelAuc(prediction, groundtrue, class_num=NUM_I, width=SIZE_I, height=SIZE_I) :
    # w * h *1 one voxel
    empty_value = -1.0
    auc_allClass = empty_value * np.ones((class_num), dtype=np.float32)
    ii = 0
    for c in label_list :
        # print('auc')
        # print(c)
        # 计算每个分类的AUC,每个分类来说是二分类 20次
        flatten_gt = np.reshape(groundtrue, (width * height, 3))
        flatten_pr = np.reshape(prediction, (width * height, 3))
        if (not ((c == flatten_gt).all(1).any()) and not ((c == flatten_pr).all(1).any())) :
            # 跳过没出现在本例子中的分类
            ii = ii + 1
            continue
        width = groundtrue.shape[0]
        height = groundtrue.shape[1]
        predictionMatrix = np.zeros((width, height))
        groundtrueMatrix = np.zeros((width, height))
        # predictionMatrixFlatten = prediction.flatten()
        # groundtrueMatrixFlatten = groundtrue.flatten()
        # print(predictionMatrixFlatten)
        # print("--------------------------------")

        for i in range(0, width) :
            for j in range(0, height) :
                if all(groundtrue[i][j] == c) :
                    groundtrueMatrix[i][j] = 1
                else :
                    groundtrueMatrix[i][j] = 0

                if all(prediction[i][j] == c) :
                    predictionMatrix[i][j] = 1
                else :
                    predictionMatrix[i][j] = 0

        # for i in range(0, len(predictionMatrixFlatten)):
        #     if predictionMatrixFlatten[i] == c:
        #         predictionMatrixFlatten[i] = 1
        #     else:
        #         predictionMatrixFlatten[i] = 0
        #
        # for i in range(0, len(groundtrueMatrixFlatten)):
        #     if groundtrueMatrixFlatten[i] == c:
        #         groundtrueMatrixFlatten[i] = 1
        #     else:
        #         groundtrueMatrixFlatten[i] = 0

        predictionMatrixFlatten = predictionMatrix.flatten()
        groundtrueMatrixFlatten = groundtrueMatrix.flatten()
        fpr, tpr, thresholds = roc_curve(groundtrueMatrixFlatten, predictionMatrixFlatten, pos_label=1)
        AUC_oneClass = auc(fpr, tpr)
        # print('AUC_oneClass',AUC_oneClass)
        auc_allClass[ii] = AUC_oneClass
        ii = ii + 1
    # print('1AUC')
    aucs = np.where(auc_allClass == -1.0, np.nan, auc_allClass)
    voxel_level_auc = np.nanmean(aucs[1 :])
    voxel_level_auc_std = np.nanstd(aucs[1 :], ddof=1)
    # print('voxel_level_auc_std',voxel_level_auc_std)
    return voxel_level_auc, voxel_level_auc_std


def oneCaseLevelAll(prediction, groundtrue, class_num=NUM_I) :
    # 输入w*h*c*3变成w*h*3
    means = {}
    stds = {}

    dices_oneCase_std = []
    accs_oneCase_std = []
    precisions_oneCase_std = []
    aucs_oneCase_std = []

    dices_oneCase_mean = []
    accs_oneCase_mean = []
    precisions_oneCase_mean = []
    aucs_oneCase_mean = []

    width = groundtrue.shape[0]
    height = groundtrue.shape[1]
    voxel = groundtrue.shape[2]

    for v in range(0, voxel) :
        # print('voxel:',v)
        print('-----processing with No ', v, ' voxel png')
        oneVoxelChip_target = np.zeros((width, height, 3))
        oneVoxelChip_prediction = np.zeros((width, height, 3))
        for i in range(0, width) :
            for j in range(0, height) :
                # 以数组赋值的
                oneVoxelChip_target[i][j] = groundtrue[i][j][v]
                oneVoxelChip_prediction[i][j] = prediction[i][j][v]

        oneVoxelAuc_mean, oneVoxelAuc_std = oneVoxeLevelAuc(oneVoxelChip_prediction, oneVoxelChip_target, class_num,
                                                            width, height)

        tp_allClass, fp_allClass, tn_allClass, fn_allClass = getOneVoxelFPTPTNFN(oneVoxelChip_prediction,
                                                                                 oneVoxelChip_target, class_num, width,
                                                                                 height)
        oneVoxelDice_mean, oneVoxelDice_std = oneVoxeLevelDice(oneVoxelChip_prediction, oneVoxelChip_target,
                                                               tp_allClass, fp_allClass,
                                                               tn_allClass, fn_allClass, class_num, width, height)
        oneVoxelAccu_mean, oneVoxelAccu_std = oneVoxeLevelAccuracy(oneVoxelChip_prediction, oneVoxelChip_target,
                                                                   tp_allClass, fp_allClass,
                                                                   tn_allClass, fn_allClass, class_num, width, height)
        oneVoxelPrecision_mean, oneVoxelPrecision_std = oneVoxeLevelPrecision(oneVoxelChip_prediction,
                                                                              oneVoxelChip_target, tp_allClass,
                                                                              fp_allClass, tn_allClass, fn_allClass,
                                                                              class_num, width, height)
        dices_oneCase_mean.append(oneVoxelDice_mean)
        accs_oneCase_mean.append(oneVoxelAccu_mean)
        precisions_oneCase_mean.append(oneVoxelPrecision_mean)
        aucs_oneCase_mean.append(oneVoxelAuc_mean)

        dices_oneCase_std.append(oneVoxelDice_std)
        accs_oneCase_std.append(oneVoxelAccu_std)
        precisions_oneCase_std.append(oneVoxelPrecision_std)
        aucs_oneCase_std.append(oneVoxelAuc_std)

    means['dices_oneCase'] = np.mean(dices_oneCase_mean)
    means['accs_oneCase'] = np.mean(accs_oneCase_mean)
    means['precisions_oneCase'] = np.mean(precisions_oneCase_mean)
    means['aucs_oneCase'] = np.mean(aucs_oneCase_mean)
    stds['dices_oneCase'] = np.mean(dices_oneCase_std)
    stds['accs_oneCase'] = np.mean(accs_oneCase_std)
    stds['precisions_oneCase'] = np.mean(precisions_oneCase_std)
    stds['aucs_oneCase'] = np.mean(aucs_oneCase_std)
    return means, stds


def evaluate_demo(prediction_allCase_folders, target_allCase_folder) :
    # 输入的是case的文件夹列表，包含多个case的文件夹的路径
    means = {}
    stds = {}

    dices_all_std = []
    accs_all_std = []
    pres_all_std = []
    aucs_all_std = []

    dices_all_mean = []
    accs_all_mean = []
    pres_all_mean = []
    aucs_all_mean = []

    floders_len = len(prediction_allCase_folders)
    for i in range(0, floders_len) :
        print('processing with No ', i + 1, ' case')
        # case_floder travel
        prediction_oneCase_floder = prediction_allCase_folders[i]
        target_oneCase_floder = target_allCase_folder[i]
        print('prediction_oneCase_floder:',prediction_oneCase_floder,'target_oneCase_floder:',target_oneCase_floder)

        # prediction_voxel0_PngPath = os.path.join(prediction_oneCase_floder, '0_predict.png')
        prediction_voxel0_PngPath = os.path.join(prediction_oneCase_floder, '0_predict.png')
        target_voxel0_PngPath = os.path.join(target_oneCase_floder, '0.png')

        prediction_case_matrix, width_p, hight_p = imgToMatrix(prediction_voxel0_PngPath)
        target_case_matrix, width_t, hight_t = imgToMatrix(target_voxel0_PngPath)
        # w*h*3
        voxel_len = len(os.listdir(target_oneCase_floder))
        for j in range(1, voxel_len) :
            # voxel_file travel and j is channel
            # prediction_voxel_PngPath=os.path.join(prediction_oneCase_floder,(str(j)+'_predict.png'))
            prediction_voxel_PngPath = os.path.join(prediction_oneCase_floder, (str(j) + '_predict.png'))
            target_voxel_PngPath = os.path.join(target_oneCase_floder, (str(j) + '.png'))

            prediction_voxel_Martrix, _, _ = imgToMatrix(prediction_voxel_PngPath)
            target_voxel_Martrix, _, _ = imgToMatrix(target_voxel_PngPath)
            # w*h*3
            prediction_case_matrix = np.concatenate((prediction_case_matrix, prediction_voxel_Martrix), axis=1)
            target_case_matrix = np.concatenate((target_case_matrix, target_voxel_Martrix), axis=1)
            # w*(voxel_len * h)*3
        # 将每个切片组合成三维矩阵，z轴代表切片数量
        prediction_case_matrix = np.reshape(prediction_case_matrix, (width_p, hight_p, voxel_len, 3), order='F')
        target_case_matrix = np.reshape(target_case_matrix, (width_t, hight_t, voxel_len, 3), order='F')
        # print(prediction_case_matrix.shape)
        means_oneCase, stds_oneCase = oneCaseLevelAll(prediction_case_matrix, target_case_matrix, class_num=NUM_I)

        dsc_oneCase_std = stds_oneCase['dices_oneCase']
        acc_oneCase_std = stds_oneCase['accs_oneCase']
        precision_oneCase_std = stds_oneCase['precisions_oneCase']
        auc_oneCase_std = stds_oneCase['aucs_oneCase']

        dsc_oneCase_mean = means_oneCase['dices_oneCase']
        acc_oneCase_mean = means_oneCase['accs_oneCase']
        precision_oneCase_mean = means_oneCase['precisions_oneCase']
        auc_oneCase_mean = means_oneCase['aucs_oneCase']

        dices_all_mean.append(dsc_oneCase_mean)
        accs_all_mean.append(acc_oneCase_mean)
        pres_all_mean.append(precision_oneCase_mean)
        aucs_all_mean.append(auc_oneCase_mean)

        dices_all_std.append(dsc_oneCase_std)
        accs_all_std.append(acc_oneCase_std)
        pres_all_std.append(precision_oneCase_std)
        aucs_all_std.append(auc_oneCase_std)
        print('case ', i + 1, ' dice: ', dsc_oneCase_mean, ' dice_std: ', dsc_oneCase_std)
        print('case ', i + 1, ' acc: ', acc_oneCase_mean, ' acc_std: ', acc_oneCase_std)
        print('case ', i + 1, ' precision: ', precision_oneCase_mean, ' precision_std: ', precision_oneCase_std)
        print('case ', i + 1, ' auc: ', auc_oneCase_mean, ' auc_std: ', auc_oneCase_std)
        print('----------------------------------------------------------')

    means['dscs'] = np.mean(dices_all_mean)
    means['accs'] = np.mean(accs_all_mean)
    means['pres'] = np.mean(pres_all_mean)
    means['aucs'] = np.mean(aucs_all_mean)

    stds['dscs'] = np.mean(dices_all_std)
    stds['accs'] = np.mean(accs_all_std)
    stds['pres'] = np.mean(pres_all_std)
    stds['aucs'] = np.mean(aucs_all_std)
    return means, stds


def main(NameOfModel):
    class Logger(object) :
        def __init__(self, filename=str(NameOfModel) + ".log") :
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message) :
            self.terminal.write(message)
            self.log.write(message)

        def flush(self) :
            pass
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('evalue/' + str(NameOfModel) + '.txt')

    prediction_allCase_folders = []
    target_allCase_folder = []
    for root, dirs, files in os.walk('./results/'+str(NameOfModel)):
        for dir in dirs:
            folder_root = os.path.join(root, dir)
            prediction_allCase_folders.append(folder_root)

    for root, dirs, files in os.walk('./data_2/label_t'):
        for dir in dirs :
            folder_root = os.path.join(root, dir)
            target_allCase_folder.append(folder_root)

    mean_all, std_all = evaluate_demo(prediction_allCase_folders, target_allCase_folder)

    std_dice = std_all['dscs']
    std_acc = std_all['accs']
    std_precision = std_all['pres']
    std_auc = std_all['aucs']

    mean_dice = mean_all['dscs']
    mean_acc = mean_all['accs']
    mean_precision = mean_all['pres']
    mean_auc = mean_all['aucs']

    print('mean_dice is ' + str(mean_dice) + '+' + str(std_dice))
    print('mean_acc is ' + str(mean_acc) + '+' + str(std_acc))
    print('mean_precision is ' + str(mean_precision) + '+' + str(std_precision))
    print('mean_auc is ' + str(mean_auc) + '+' + str(std_auc))


if __name__ == "__main__" :
    NameOfModel = 'LSWNet_1'
    main(NameOfModel)
