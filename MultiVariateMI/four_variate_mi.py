import sys
import torch
import random


def four_variate_IID_loss(x_1, x_2, x_3, x_4, EPS=sys.float_info.epsilon):
  k = 10  # had softmax applied

  #joint_probability_1_2_3_4 = joint_probability_8(x_1, x_2, x_3, x_4, x_5, x_6)
  joint_probability_1_2_3_4 = joint_probability(x_1, x_2, x_3, x_4)

  # assert (joint_probability_1_2_3_4.size() == (k, k, k, k, k, k))

  # mvmi = multi_variate_mutual_info(joint_probability_1_2_3_4)
  total_corr = total_correlation(joint_probability_1_2_3_4)
  loss = my_loss(joint_probability_1_2_3_4)

  if random.uniform(0,1)>0.995:
      print("total corr: ", total_corr)

  loss = loss.sum() + total_corr

  return loss


def six_variate_IID_loss(x_1, x_2, x_3, x_4, x_5, x_6, EPS=sys.float_info.epsilon):
  k = 10  # had softmax applied

  joint_probability_1_2_3_4 = joint_probability_8(x_1, x_2, x_3, x_4, x_5, x_6)
  #joint_probability_1_2_3_4 = joint_probability(x_1, x_2, x_3, x_4)

  # assert (joint_probability_1_2_3_4.size() == (k, k, k, k, k, k))

  # mvmi = multi_variate_mutual_info(joint_probability_1_2_3_4)

  loss = my_loss_8(joint_probability_1_2_3_4)

  loss = loss.sum()

  return loss


def my_loss_8(joint_probability_1_2_3_4):
    p_1, p_2, p_3, p_4 = one_variate_marginals(joint_probability_1_2_3_4)
    classes = 10
    general_target = 1 / classes


    sum = 0
    for num in range(10):
        joint_prob = joint_probability_1_2_3_4[num, num, num, num, num, num]

        sum += - torch.log(joint_prob)  # diag_element(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4, number)

    return sum

def my_loss(joint_probability_1_2_3_4):
    #p_1, p_2, p_3, p_4 = one_variate_marginals(joint_probability_1_2_3_4)
    classes = 10
    class_weight = 1/classes


    sum = 0
    for num in range(10):
        joint_prob = joint_probability_1_2_3_4[num, num, num, num]

        if random.uniform(0, 1) > 0.999:
            print("diag joint ", num, joint_prob)

        # prob = p_1[num, num, num, num] * p_2[num, num, num, num] * p_3[num, num, num, num] * p_4[num, num, num, num]
        #
        # target = torch.log(1 - torch.abs(joint_prob-prob))
        # t1 = torch.log(1 - torch.log(p_1[num, num, num, num] - general_target))
        # t2 = torch.log(1 - torch.log(p_2[num, num, num, num] - general_target))
        # t3 = torch.log(1 - torch.log(p_3[num, num, num, num] - general_target))
        # t4 = torch.log(1 - torch.log(p_4[num, num, num, num] - general_target))

        #joint_target = torch.log(1 - torch.abs(joint_prob - general_target))
        # log_1 = torch.log(p_1[num, num, num, num])
        # log_2 = torch.log(p_2[num, num, num, num])
        # log_3 = torch.log(p_3[num, num, num, num])
        # log_4 = torch.log(p_4[num, num, num, num])

        sum += -torch.log(joint_prob)

    return sum


def diag_element(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4, num):
    # prob = torch.log(joint_probability_1_2_3_4[num, num, num, num]) - \
    #        torch.log(p_1[num, num, num, num]) - \
    #        torch.log(p_2[num, num, num, num]) - \
    #        torch.log(p_3[num, num, num, num]) - \
    #        torch.log(p_4[num, num, num, num])

    #
    # normalised_prob = joint_probability_1_2_3_4[num, num, num, num] / \
    #                   (p_1[num, num, num, num] *
    #                    p_2[num, num, num, num] *
    #                    p_3[num, num, num, num] *
    #                    p_4[num, num, num, num])

    # print(torch.log(joint_probability_1_2_3_4[num, num, num, num]))
    # # print(p_1[num, num, num, num], p_2[num, num, num, num], p_3[num, num, num, num], p_4[num, num, num, num])
    # # print(normalised_prob)
    # # print(torch.log(normalised_prob))
    # print()
    return - torch.log(joint_probability_1_2_3_4[num, num, num, num]) #* torch.log(normalised_prob)

def total_correlation(joint_probability_1_2_3_4):
    p_1, p_2, p_3, p_4 = one_variate_marginals(joint_probability_1_2_3_4)

    numerator = torch.log(joint_probability_1_2_3_4)
    denominator = torch.log(p_1) + torch.log(p_2) + torch.log(p_3) + torch.log(p_4)

    total_corr = - joint_probability_1_2_3_4 * (numerator - denominator) #* (-denominator)
    total_corr = total_corr.sum()

    return total_corr


def multi_variate_mutual_info(joint_probability_1_2_3_4):
    '''
    '''

    p_1, p_2, p_3, p_4 = one_variate_marginals(joint_probability_1_2_3_4)
    p_1_2, p_1_3, p_1_4, p_2_3, p_2_4, p_3_4 = two_variate_marginals(joint_probability_1_2_3_4)
    p_1_2_3, p_1_2_4, p_1_3_4, p_2_3_4 = three_variate_marginals(joint_probability_1_2_3_4)

    numerator = torch.log(joint_probability_1_2_3_4) + \
                torch.log(p_1_2) + \
                torch.log(p_1_3) + \
                torch.log(p_1_4) + \
                torch.log(p_2_3) + \
                torch.log(p_2_4) + \
                torch.log(p_3_4)

    denominator = torch.log(p_1_2_3) + \
                  torch.log(p_1_2_4) + \
                  torch.log(p_1_3_4) + \
                  torch.log(p_2_3_4) + \
                  torch.log(p_1) + \
                  torch.log(p_2) + \
                  torch.log(p_3) + \
                  torch.log(p_4)

    multi_variate_mi = - joint_probability_1_2_3_4 * (numerator - denominator)

    return multi_variate_mi

def reverse_total_crrelation(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4):
  numerator = torch.log(p_1) + torch.log(p_2) + torch.log(p_3) + torch.log(p_4)
  denominator = torch.log(joint_probability_1_2_3_4)

  rev = (- (joint_probability_1_2_3_4) * (numerator - denominator))

  return rev


def dual_total_correlation(joint_probability_1_2_3_4, p_1, p_2, p_3, p_4, marginal_1, marginal_2, marginal_3, marginal_4):
    joint_entr = joint_entropy(joint_probability_1_2_3_4)

    conditional_entropy_1 = -joint_probability_1_2_3_4 * torch.log(p_1)
    conditional_entropy_2 = -joint_probability_1_2_3_4 * torch.log(p_2)
    conditional_entropy_3 = -joint_probability_1_2_3_4 * torch.log(p_3)
    conditional_entropy_4 = -joint_probability_1_2_3_4 * torch.log(p_4)

    # conditional_entropy_1 = joint_entr - (- p_1 * torch.log(p_1))
    # conditional_entropy_2 = joint_entr - (- p_2 * torch.log(p_2))
    # conditional_entropy_3 = joint_entr - (- p_3 * torch.log(p_3))
    # conditional_entropy_4 = joint_entr - (- p_4 * torch.log(p_4))

    conditional_entropies = conditional_entropy_1 + conditional_entropy_2 + conditional_entropy_3 + conditional_entropy_4

    dtc = joint_entr - conditional_entropies
    normalised_dtc = dtc / joint_entr

    return normalised_dtc


def joint_entropy(joint_probability):
  '''
  joint entropy is a measure of the uncertainty associated with a set of variables.
  :return:
  '''

  return - joint_probability * torch.log(joint_probability)


def Information_Quality_Ratio(multi_variate_mi, joint_entropy):
  '''
  This normalized version also known as Information Quality Ratio (IQR)
   which quantifies the amount of information of a variable based on another variable against total uncertainty
  :param multi_variate_mi:
  :param joint_entropy:
  :return:
  '''
  return multi_variate_mi / joint_entropy

def joint_probability_8(x_1, x_2, x_3, x_4, x_5, x_6):
    # produces variable that requires grad (since args require grad)

    bn, k = x_1.size()
    assert (x_2.size(0) == bn and x_2.size(1) == k)
    assert (x_3.size(1) == k and x_4.size(1) == k)
    assert (x_5.size(1) == k and x_6.size(1) == k)

    # print("x1", x_1.shape)
    # print("x1", x_1.unsqueeze(2).shape)
    # print("")
    # print("x2", x_2.shape)
    # print("x2", x_2.unsqueeze(1).shape)
    # print("")

    combine_1_2 = x_1.unsqueeze(2) * x_2.unsqueeze(1)  # batch, k, k

    x_3_unsq = x_3.unsqueeze(1).unsqueeze(2)
    combine_1_2_3 = combine_1_2.unsqueeze(3) * x_3_unsq

    x_4_unsq = x_4.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    combine_1_2_3_4 = combine_1_2_3.unsqueeze(4) * x_4_unsq

    x_5_unsq = x_5.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    combine_1_2_3_4_5 = combine_1_2_3_4.unsqueeze(5) * x_5_unsq

    x_6_unsq = x_6.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)
    combine_1_2_3_4_5_6 = combine_1_2_3_4_5.unsqueeze(6) * x_6_unsq

    # x_7_unsq = x_7.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
    # combine_1_2_3_4_5_6_7 = combine_1_2_3_4_5_6.unsqueeze(7) * x_7_unsq
    #
    # x_8_unsq = x_8.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6).unsqueeze(7)
    # combine_1_2_3_4_5_6_7_8 = combine_1_2_3_4_5_6_7.unsqueeze(8) * x_8_unsq

    combine_1_2_3_4_5_6 = combine_1_2_3_4_5_6.sum(dim=0)  # k, k, k, k
    combine_1_2_3_4_5_6 = combine_1_2_3_4_5_6 / combine_1_2_3_4_5_6.sum()  # normalise

    EPS = sys.float_info.epsilon
    combine_1_2_3_4_5_6[combine_1_2_3_4_5_6 < EPS] = EPS

    return combine_1_2_3_4_5_6

def joint_probability(x_1, x_2, x_3, x_4):
  # produces variable that requires grad (since args require grad)

  bn, k = x_1.size()
  assert (x_2.size(0) == bn and x_2.size(1) == k)
  assert (x_3.size(1) == k and x_4.size(1) == k)

  # print("x1", x_1.shape)
  # print("x1", x_1.unsqueeze(2).shape)
  # print("")
  # print("x2", x_2.shape)
  # print("x2", x_2.unsqueeze(1).shape)
  # print("")

  combine_1_2 = x_1.unsqueeze(2) * x_2.unsqueeze(1)  # batch, k, k

  x_3_unsq = x_3.unsqueeze(1).unsqueeze(2)

  combine_1_2_3 = combine_1_2.unsqueeze(3) * x_3_unsq
  x_4_unsq = x_4.unsqueeze(1).unsqueeze(2).unsqueeze(3)

  combine_1_2_3_4 = combine_1_2_3.unsqueeze(4) * x_4_unsq
  combine_1_2_3_4 = combine_1_2_3_4.sum(dim=0)  # k, k, k, k
  combine_1_2_3_4 = combine_1_2_3_4 / combine_1_2_3_4.sum()  # normalise

  EPS = sys.float_info.epsilon
  combine_1_2_3_4[combine_1_2_3_4 < EPS] = EPS

  return combine_1_2_3_4


def three_variate_marginals(joint_probability_1_2_3_4, k=10, EPS=sys.float_info.epsilon):
    marginal_4 = joint_probability_1_2_3_4.sum(dim=3).view(k, k, k, 1).expand(k, k, k, k)
    marginal_3 = joint_probability_1_2_3_4.sum(dim=2).view(k, k, 1, k).expand(k, k, k, k)
    marginal_2 = joint_probability_1_2_3_4.sum(dim=1).view(k, 1, k, k).expand(k, k, k, k)
    marginal_1 = joint_probability_1_2_3_4.sum(dim=0).view(1, k, k, k).expand(k, k, k, k)

    marginal_4[(marginal_4 < EPS).data] = EPS
    marginal_3[(marginal_3 < EPS).data] = EPS
    marginal_2[(marginal_2 < EPS).data] = EPS
    marginal_1[(marginal_1 < EPS).data] = EPS

    return marginal_1, marginal_2, marginal_3, marginal_4

def two_variate_marginals(joint_probability_1_2_3_4, k=10, EPS=sys.float_info.epsilon):
    p_1_2 = joint_probability_1_2_3_4.sum(dim=3).sum(dim=2).view(k, k, 1, 1).expand(k, k, k, k)
    p_1_3 = joint_probability_1_2_3_4.sum(dim=3).sum(dim=1).view(k, 1, k, 1).expand(k, k, k, k)
    p_1_4 = joint_probability_1_2_3_4.sum(dim=2).sum(dim=1).view(k, 1, 1, k).expand(k, k, k, k)

    p_2_3 = joint_probability_1_2_3_4.sum(dim=3).sum(dim=0).view(1, k, k, 1).expand(k, k, k, k)
    p_2_4 = joint_probability_1_2_3_4.sum(dim=2).sum(dim=0).view(1, k, 1, k).expand(k, k, k, k)

    p_3_4 = joint_probability_1_2_3_4.sum(dim=1).sum(dim=0).view(1, 1, k, k).expand(k, k, k, k)

    p_1_2[(p_1_2 < EPS).data] = EPS
    p_1_3[(p_1_3 < EPS).data] = EPS
    p_1_4[(p_1_4 < EPS).data] = EPS

    p_2_3[(p_2_3 < EPS).data] = EPS
    p_2_4[(p_2_4 < EPS).data] = EPS

    p_3_4[(p_3_4 < EPS).data] = EPS

    return p_1_2, p_1_3, p_1_4, p_2_3, p_2_4, p_3_4

def one_variate_marginals(joint_probability_1_2_3_4, k=10, EPS=sys.float_info.epsilon):
    p_1 = joint_probability_1_2_3_4.sum(dim=1).sum(dim=1).sum(dim=1).view(k, 1, 1, 1).expand(k, k, k, k)
    p_2 = joint_probability_1_2_3_4.sum(dim=0).sum(dim=1).sum(dim=1).view(1, k, 1, 1).expand(k, k, k, k)
    p_3 = joint_probability_1_2_3_4.sum(dim=0).sum(dim=0).sum(dim=1).view(1, 1, k, 1).expand(k, k, k, k)
    p_4 = joint_probability_1_2_3_4.sum(dim=0).sum(dim=0).sum(dim=0).view(1, 1, 1, k).expand(k, k, k, k)

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway

    joint_probability_1_2_3_4[joint_probability_1_2_3_4 < EPS] = EPS
    p_1[(p_1 < EPS).data] = EPS
    p_2[(p_2 < EPS).data] = EPS
    p_3[(p_3 < EPS).data] = EPS
    p_4[(p_4 < EPS).data] = EPS

    return p_1, p_2, p_3, p_4