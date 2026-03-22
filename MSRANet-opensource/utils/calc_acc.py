import torch


def calc_acc(logits, label, ignore_index=-100, mode="multiclass"):
    if mode == "binary":
        indices = torch.round(logits).type(label.type())    #四舍五入到最近的整数
    elif mode == "multiclass":
        indices = torch.max(logits, dim=1)[1]   #1维上的最大值索引

    if label.size() == logits.size():
        ignore = 1 - torch.round(label.sum(dim=1))  #属于多个类别会被忽略掉
        label = torch.max(label, dim=1)[1]
    else:
        ignore = torch.eq(label, ignore_index).view(-1) #忽略标签索引为-100的样本

    correct = torch.eq(indices, label).view(-1)
    num_correct = torch.sum(correct)
    num_examples = logits.shape[0] - ignore.sum()

    return num_correct.float() / num_examples.float()
