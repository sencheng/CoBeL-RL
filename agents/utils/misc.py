def Get_Single_Input(in_,shape):
    if in_[0].shape == shape:
        return in_[0]
    else:
        return in_[0][0]

def Average(lst):
    return sum(lst) / len(lst)