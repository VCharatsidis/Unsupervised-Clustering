import sys
import torch
import random


def three_variate_IID_loss(x_1, x_2, x_3, EPS=sys.float_info.epsilon):
  # has had softmax applied
  k = 10
  joint_1_2_3 = joint(x_1, x_2, x_3)
  assert (joint_1_2_3.size() == (k, k, k))

  p_i = joint_1_2_3.sum(dim=1).sum(dim=1).view(k, 1, 1).expand(k, k, k)
  p_j = joint_1_2_3.sum(dim=0).sum(dim=1).view(1, k, 1).expand(k, k, k)
  p_z = joint_1_2_3.sum(dim=0).sum(dim=0).view(1, 1, k).expand(k, k, k)

  # print("p")
  # print(p_i.shape)
  #
  # input()

  # print(joint_1_2_3.sum(dim=2))
  # print(joint_1_2_3.sum(dim=2).view(k, k, 1).shape)
  # input()
  p_i_j = joint_1_2_3.sum(dim=2).view(k, k, 1).expand(k, k, k)
  p_i_z = joint_1_2_3.sum(dim=1).view(k, 1, k).expand(k, k, k)
  p_j_z = joint_1_2_3.sum(dim=0).view(1, k, k).expand(k, k, k)

  # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
  # print(joint_1_2_3)
  # print(joint_1_2_3.shape)
  # print(joint_1_2_3[0][0].shape)
  # joint_1_2_3[0][joint_1_2_3[0] < EPS] = EPS
  # joint_1_2_3[:, (joint_1_2_3 < EPS).data, :] = EPS
  # joint_1_2_3[(joint_1_2_3 < EPS).data, :, :] = EPS
  # p_j[(p_j < EPS).data] = EPS
  # p_i[(p_i < EPS).data] = EPS
  # p_z[(p_z < EPS).data] = EPS

  numerator = torch.log(p_i_j) + torch.log(p_i_z) + torch.log(p_j_z)
  denominator = torch.log(joint_1_2_3) + torch.log(p_i) + torch.log(p_j) + torch.log(p_z)

  # Total correlation
  # numerator = torch.log(joint_1_2_3)
  # denominator = torch.log(p_i) + torch.log(p_j) + torch.log(p_z)

  loss = - joint_1_2_3 * (numerator - denominator)
  loss = loss.sum()
  #loss = torch.abs(loss)
  return loss


def joint(x_1, x_2, x_3):
  # produces variable that requires grad (since args require grad)

  bn, k = x_1.size()
  assert (x_2.size(0) == bn and x_2.size(1) == k)
  assert (x_3.size(1) == k)

  # print("x1", x_1.shape)
  # print("x1", x_1.unsqueeze(2).shape)
  # print("")
  # print("x2", x_2.shape)
  # print("x2", x_2.unsqueeze(1).shape)
  # print("")

  combine_1_2 = x_1.unsqueeze(2) * x_2.unsqueeze(1)  # batch, k, k
  x_3_unsq = x_3.unsqueeze(1).unsqueeze(2)

  combine_1_2_3 = combine_1_2.unsqueeze(3) * x_3_unsq

  combine_1_2_3 = combine_1_2_3.sum(dim=0)  # k, k, k
  combine_1_2_3 = combine_1_2_3 / combine_1_2_3.sum()  # normalise

  return combine_1_2_3


def compute_joint(x_out, x_tf_out):
  # produces variable that requires grad (since args require grad)

  bn, k = x_out.size()
  assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

  p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
  p_i_j = p_i_j.sum(dim=0)  # k, k
  #p_i_j = (p_i_j + p_i_j.t()) / 2.
  p_i_j = p_i_j / p_i_j.sum()  # normalise

  return p_i_j