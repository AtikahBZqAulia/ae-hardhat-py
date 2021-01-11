def CG_MNIST(VV, Dim, X2):
    import numpy as np
    L1 = Dim[0]
    L2 = Dim[1]
    L3 = Dim[2]
    L4 = Dim[3]
    L5 = Dim[4]
    L6 = Dim[5]
    L7 = Dim[6]
    L8 = Dim[7]
    L9 = Dim[8]
    L10 = Dim[9]
    L11 = Dim[10]
    L12 = Dim[11]
    L13 = Dim[12]
    L14 = Dim[13]
    L15 = Dim[14]
    L16 = Dim[15]
    L17 = Dim[16]
    N = X2.shape[0]

    # Do decomversion
    W1 = VV[0][0:(L1[0]+1)*L2[0]].reshape(L1[0]+1,L2[0])
    X3 = (L1[0]+1)*L2[0]
    W2 = VV[0][X3:X3+(L2[0]+1)*L3[0]].reshape(L2[0]+1, L3[0])
    X3 = X3+(L2[0]+1)*L3[0]
    W3 = VV[0][X3:X3+(L3[0]+1)*L4[0]].reshape(L3[0]+1, L4[0])
    X3 = X3+(L3[0]+1)*L4[0]
    W4 = VV[0][X3:X3+(L4[0]+1)*L5[0]].reshape(L4[0]+1, L5[0])
    X3 = X3+(L4[0]+1)*L5[0]
    W5 = VV[0][X3:X3+(L5[0]+1)*L6[0]].reshape(L5[0]+1, L6[0])
    X3 = X3+(L5[0]+1)*L6[0]
    W6 = VV[0][X3:X3+(L6[0]+1)*L7[0]].reshape(L6[0]+1, L7[0])
    X3 = X3+(L6[0]+1)*L7[0]
    W7 = VV[0][X3:X3+(L7[0]+1)*L8[0]].reshape(L7[0]+1, L8[0])
    X3 = X3+(L7[0]+1)*L8[0]
    W8 = VV[0][X3:X3+(L8[0]+1)*L9[0]].reshape(L8[0]+1, L9[0])

    X3 = X3+(L8[0]+1)*L9[0]
    W9 = VV[0][X3:X3+(L9[0]+1)*L10[0]].reshape(L9[0]+1, L10[0])
    X3 = X3+(L9[0]+1)*L10[0]
    W10 = VV[0][X3:X3+(L10[0]+1)*L11[0]].reshape(L10[0]+1, L11[0])
    X3 = X3+(L10[0]+1)*L11[0]
    W11 = VV[0][X3:X3+(L11[0]+1)*L12[0]].reshape(L11[0]+1, L12[0])
    X3 = X3+(L11[0]+1)*L12[0]
    W12 = VV[0][X3:X3+(L12[0]+1)*L13[0]].reshape(L12[0]+1, L13[0])
    X3 = X3+(L12[0]+1)*L13[0]
    W13 = VV[0][X3:X3+(L13[0]+1)*L14[0]].reshape(L13[0]+1, L14[0])
    X3 = X3+(L13[0]+1)*L14[0]
    W14 = VV[0][X3:X3+(L14[0]+1)*L15[0]].reshape(L14[0]+1, L15[0])
    X3 = X3+(L14[0]+1)*L15[0]
    W15 = VV[0][X3:X3+(L15[0]+1)*L16[0]].reshape(L15[0]+1, L16[0])
    X3 = X3+(L15[0]+1)*L16[0]
    W16 = VV[0][X3:X3+(L16[0]+1)*L17[0]].reshape(L16[0]+1, L17[0])

    X2 = np.append(X2, np.ones((N, 1)), axis=1)
    W1_PROBS = 1.0/(1 + np.exp(np.matmul(-X2, W1)))
    W1_PROBS = np.append(W1_PROBS, np.ones((N, 1)), axis=1)
    W2_PROBS = 1.0/(1 + np.exp(np.matmul(-W1_PROBS, W2)))
    W2_PROBS = np.append(W2_PROBS, np.ones((N, 1)), axis=1)
    W3_PROBS = 1.0/(1 + np.exp(np.matmul(-W2_PROBS,W3)))
    W3_PROBS = np.append(W3_PROBS, np.ones((N, 1)), axis=1)
    W4_PROBS = 1.0/(1 + np.exp(np.matmul(W3_PROBS, W4)))
    W4_PROBS = np.append(W4_PROBS, np.ones((N, 1)), axis=1)
    W5_PROBS = 1.0/(1 + np.exp(np.matmul(-W4_PROBS,W5)))
    W5_PROBS = np.append(W5_PROBS, np.ones((N, 1)), axis=1)
    W6_PROBS = 1.0/(1 + np.exp(np.matmul(-W5_PROBS, W6)))
    W6_PROBS = np.append(W6_PROBS, np.ones((N, 1)), axis=1)
    W7_PROBS = 1.0/(1 + np.exp(np.matmul(-W6_PROBS, W7)))
    W7_PROBS = np.append(W7_PROBS, np.ones((N, 1)), axis=1)
    
    W8_PROBS = np.matmul(W7_PROBS, W8)
    W8_PROBS = np.append(W8_PROBS, np.ones((N, 1)), axis=1)

    W9_PROBS = 1.0/(1 + np.exp(np.matmul(-W8_PROBS,W9)))
    W9_PROBS = np.append(W9_PROBS, np.ones((N, 1)), axis=1)
    W10_PROBS = 1.0/(1 + np.exp(np.matmul(-W9_PROBS, W10)))
    W10_PROBS = np.append(W10_PROBS, np.ones((N, 1)), axis=1)
    W11_PROBS = 1.0/(1 + np.exp(np.matmul(-W10_PROBS, W11)))
    W11_PROBS = np.append(W11_PROBS, np.ones((N, 1)), axis=1)
    W12_PROBS = 1.0/(1 + np.exp(np.matmul(-W11_PROBS,W12)))
    W12_PROBS = np.append(W12_PROBS, np.ones((N, 1)), axis=1)
    W13_PROBS = 1.0/(1 + np.exp(np.matmul(-W12_PROBS, W13)))
    W13_PROBS = np.append(W13_PROBS, np.ones((N, 1)), axis=1)
    W14_PROBS = 1.0/(1 + np.exp(np.matmul(-W13_PROBS, W14)))
    W14_PROBS = np.append(W14_PROBS, np.ones((N, 1)), axis=1)
    W15_PROBS = 1.0/(1 + np.exp(np.matmul(-W14_PROBS, W15)))
    W15_PROBS = np.append(W15_PROBS, np.ones((N, 1)), axis=1)

    X2_OUT = 1.0/(1 + np.exp(np.matmul(-W15_PROBS, W16)))

    f = -1/N*np.sum(np.sum(X2[:,:-1]*np.log(X2_OUT)+ (1-X2[:, :-1])*np.log(1-X2_OUT)))

    IO = 1/N*(X2_OUT-X2[:,:-1])
    Ix16 = IO
    DW16 = np.matmul(W15_PROBS.T,Ix16)

    Ix15 = (np.matmul(Ix16,W16.T))*W15_PROBS*(1-W15_PROBS)
    Ix15 = Ix15[:, :-1]
    DW15 = np.matmul(W14_PROBS.T,Ix15)

    Ix14 = (np.matmul(Ix15,W15.T))*W14_PROBS*(1-W14_PROBS)
    Ix14 = Ix14[:, :-1]
    DW14 = np.matmul(W13_PROBS.T,Ix14)

    Ix13 = (np.matmul(Ix14,W14.T))*W13_PROBS*(1-W13_PROBS)
    Ix13 = Ix13[:, :-1]
    DW13 = np.matmul(W12_PROBS.T,Ix13)

    Ix12 = (np.matmul(Ix13,W13.T))*W12_PROBS*(1-W12_PROBS)
    Ix12 = Ix12[:, :-1]
    DW12 = np.matmul(W11_PROBS.T,Ix12)

    Ix11 = (np.matmul(Ix12,W12.T))*W11_PROBS*(1-W11_PROBS)
    Ix11 = Ix11[:, :-1]
    DW11 = np.matmul(W10_PROBS.T,Ix11)

    Ix10 = (np.matmul(Ix11,W11.T))*W10_PROBS*(1-W10_PROBS)
    Ix10 = Ix10[:, :-1]
    DW10 = np.matmul(W9_PROBS.T,Ix10)

    Ix9 = (np.matmul(Ix10,W10.T))*W9_PROBS*(1-W9_PROBS)
    Ix9 = Ix9[:, :-1]
    DW9 = np.matmul(W8_PROBS.T,Ix9)

    Ix8 = (np.matmul(Ix9,W9.T))*W8_PROBS*(1-W8_PROBS)
    Ix8 = Ix8[:, :-1]
    DW8 = np.matmul(W7_PROBS.T,Ix8)

    Ix7 = (np.matmul(Ix8,W8.T))*W7_PROBS*(1-W7_PROBS)
    Ix7 = Ix7[:, :-1]
    DW7 = np.matmul(W6_PROBS.T,Ix7)

    Ix6 = (np.matmul(Ix7, W7.T))*W6_PROBS*(1-W6_PROBS)
    Ix6 = Ix6[:, :-1]
    DW6 = np.matmul(W5_PROBS.T,Ix6)

    Ix5 = (np.matmul(Ix6, W6.T))*W5_PROBS*(1-W5_PROBS)
    Ix5 = Ix5[:, :-1]
    DW5 = np.matmul(W4_PROBS.T,Ix5)

    Ix4 = np.matmul(Ix5, W5.T)
    Ix4 = Ix4[:, :-1]
    DW4 = np.matmul(W3_PROBS.T,Ix4)

    Ix3 = np.matmul(Ix4, W4.T)*W3_PROBS*(1-W3_PROBS)
    Ix3 = Ix3[:, :-1]
    DW3 = np.matmul(W2_PROBS.T,Ix3)

    Ix2 = np.matmul(Ix3, W3.T)*W2_PROBS*(1-W2_PROBS)
    Ix2 = Ix2[:, :-1]
    DW2 = np.matmul(W1_PROBS.T, Ix2)

    Ix1 = np.matmul(Ix2, W2.T)*W1_PROBS*(1-W1_PROBS)
    Ix1 = Ix1[:, :-1]
    DW1 = np.matmul(X2.T, Ix1)

    # df = lst.extend([DW1[:].T, DW2[:].T, DW3[:].T, DW4[:].T, DW5[:].T, DW6[:].T, DW7[:].T, DW8[:].T]).T
    df = np.concatenate((DW1.reshape(1,-1), DW2.reshape(1,-1), DW3.reshape(1,-1),
                DW4.reshape(1,-1), DW5.reshape(1,-1), DW6.reshape(1, -1), DW7.reshape(1, -1),
                DW8.reshape(1, -1), DW9.reshape(1,-1), DW10.reshape(1,-1), DW11.reshape(1,-1),
                DW12.reshape(1,-1), DW13.reshape(1,-1), DW14.reshape(1, -1), DW15.reshape(1, -1),
                DW16.reshape(1, -1)), axis=1)
    return f, df