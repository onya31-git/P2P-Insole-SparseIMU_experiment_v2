# test.py の予測結果を CSV に書き出すユーティリティ

import numpy as np
import pandas as pd

def export_predictions_to_csv(all_pred_pos, all_gt_pos, filename='predictions.csv'):
    """
    Exports predicted and ground truth positions to a CSV file.
    """
    pred_np = np.array(all_pred_pos)
    gt_np = np.array(all_gt_pos)
    
    flat_pred = pred_np.reshape(-1, 18, 3)
    flat_gt = gt_np.reshape(-1, 18, 3)
    
    flat_pred_2d = flat_pred.reshape(-1, 54)
    flat_gt_2d = flat_gt.reshape(-1, 54)
    
    joint_names = [
        "CenterOfGravity", "SpinePosition", "Spine4Position", "HeadPosition",
        "LShoulderPositions", "LElbowPositions", "LWristPositions",
        "RShoulderPositions", "RElbowPositions", "RWristPositions",
        "LHipPositions", "LKneePositions", "LAnklePositions", "LToePositions",
        "RHipPositions", "RKneePositions", "RAnklePositions", "RToePositions"
    ]
    
    columns = []
    # Add Pred columns
    for j in joint_names:
        columns.extend([f"Pred_{j}_X", f"Pred_{j}_Y", f"Pred_{j}_Z"])
    # Add GT columns
    for j in joint_names:
        columns.extend([f"GT_{j}_X", f"GT_{j}_Y", f"GT_{j}_Z"])
        
    combined = np.hstack((flat_pred_2d, flat_gt_2d))
    df = pd.DataFrame(combined, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")

def export_rotations_to_csv(all_pred_rot, all_gt_rot, filename='predictions_rot.csv'):
    """
    Exports predicted and ground truth rotations (6D or Euler) to a CSV file.
    """
    pred_np = np.array(all_pred_rot)
    gt_np = np.array(all_gt_rot)
    
    # [N, 18, 6]
    flat_pred = pred_np.reshape(-1, 18, 6)
    flat_gt = gt_np.reshape(-1, 18, 6)
    
    flat_pred_2d = flat_pred.reshape(-1, 108) # 18 * 6
    flat_gt_2d = flat_gt.reshape(-1, 108)
    
    joint_names = [
        "CenterOfGravity", "SpinePosition", "Spine4Position", "HeadPosition",
        "LShoulderPositions", "LElbowPositions", "LWristPositions",
        "RShoulderPositions", "RElbowPositions", "RWristPositions",
        "LHipPositions", "LKneePositions", "LAnklePositions", "LToePositions",
        "RHipPositions", "RKneePositions", "RAnklePositions", "RToePositions"
    ]
    
    columns = []
    # Add Pred columns
    for j in joint_names:
        for i in range(6):
            columns.append(f"RotPred_{j}_D{i}")
    # Add GT columns
    for j in joint_names:
        for i in range(6):
            columns.append(f"RotGT_{j}_D{i}")
            
    combined = np.hstack((flat_pred_2d, flat_gt_2d))
    df = pd.DataFrame(combined, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Saved rotations to {filename}")

def export_contacts_to_csv(all_pred_contact, all_gt_contact, filename='predictions_contact.csv'):
    """
    接地の予測確率 (sigmoid 後) と正解ラベルを CSV に書き出す。
    列: Pred_L, Pred_R (0〜1), GT_L, GT_R (0 or 1)
    """
    pred = np.array(all_pred_contact).reshape(-1, 2)
    gt = np.array(all_gt_contact).reshape(-1, 2)
    df = pd.DataFrame(np.hstack((pred, gt)), columns=['Pred_L', 'Pred_R', 'GT_L', 'GT_R'])
    df.to_csv(filename, index=False)
    print(f"Saved contacts to {filename}")
