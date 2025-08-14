import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def e1():
    """No code

    Reading documentation on:
    1. torch.Tensor()
    2. torch.cuda
    """

def e2():
    """Random tensor

    """
    random_tensor = torch.rand(7, 7)
    print("Shape: ", random_tensor.shape)


def e3():
    rt1 = torch.rand(7, 7)
    rt2 = torch.rand(1, 7)
    rt2_t = rt2.transpose(0, 1)
    mat_multi = torch.matmul(rt1, rt2_t)

    print(f"Matrix multiplication {mat_multi=}",
          f"With shape {mat_multi.shape=}")


def e4():
    torch.random.manual_seed(0)
    rt1 = torch.rand(size=(7, 7))
    rt2 = torch.rand(size=(1, 7))
    rt2_t = rt2.transpose(0, 1)
    mat_multi = torch.matmul(rt1, rt2_t)
    print(f"Matrix multiplication {mat_multi=}",
          f"With shape {mat_multi.shape=}")


def e5():
    print(f"CUDA: {torch.cuda.is_available()=}")


def e6():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)
    else:
        torch.random.manual_seed(1234)

    rt1 = torch.rand(2, 3)
    rt2 = torch.rand(2, 3)
    print(f"{rt1=}")
    print(f"{rt2=}")

    return rt1, rt2


def e7(rt1: torch.Tensor, rt2: torch.Tensor):
    rt2_t = rt2.transpose(0, 1)
    mat_multi = torch.matmul(rt1, rt2_t)
    return mat_multi
    print(f"{mat_multi=}")


def e8(mat_multi: torch.Tensor):
    mx = mat_multi.max()
    mn = mat_multi.min()
    print(f"{mat_multi=}\n max: {mx=} \n min: {mn=}")


def e9(mat_multi: torch.Tensor):
    argmx = mat_multi.argmax()
    argmn = mat_multi.argmin()
    print(f"{mat_multi=}\n argmax: {argmx=} \n argmin: {argmn=}")


def e10():
    torch.random.manual_seed(7)
    rt1 = torch.rand(size=(1, 1, 1, 10))
    rt2 = rt1[0, 0, 0]

    print(rt1, rt1.shape)
    print(rt2, rt2.shape)


e1()
e2()
e3()
e4()
e5()
rt1, rt2 = e6()
mat_multi = e7(rt1=rt1, rt2=rt2)
e8(mat_multi=mat_multi)
e9(mat_multi=mat_multi)
e10()
