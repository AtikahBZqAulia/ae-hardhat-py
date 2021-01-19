def backprop(VISHID, VISBIASES, PENRECBIASES, PENRECBIASES2, PENRECBIASES3, PENRECBIASES4, PENRECBIASES5, PENRECBIASES6, HIDRECBIASES, HIDPEN, HIDPEN2, HIDPEN3, HIDPEN4, HIDPEN5, HIDPEN6, HIDGENBIASES, HIDGENBIASES2, HIDGENBIASES3, HIDGENBIASES4, HIDGENBIASES5, HIDGENBIASES6, HIDTOP, TOPRECBIASES, TOPGENBIASES):
    import numpy as np
    import scipy.io as sio
    import scipy.optimize as sciop
    from makebatches import makebatches
    from mnistdisp import mnistdisp
    from minimize import minimize
    from CG_MNIST import CG_MNIST
    import cv2 as cv    
    import numpy as np
    import os

    MAX_EPOCH = 200
    print('Fine-tuning deep autoencoder by minimizing cross entropy error.')
    print('60 batches of 1000 cases each.')

    # file input dataset training
    path = "C:/Users/LENOVO/Downloads/data75"
    directory = os.listdir(path)   

    def get_dataset(filename):
        frame = cv.imread(path+'/'+filename)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return frame, gray

    BATCH_DATA = np.empty([0, 5625])
    i= 1
    for filename in directory:
        print(str(i) + "----- " +filename)
        frame, gray = get_dataset(filename)
        BATCH_DATA = np.append(BATCH_DATA, np.ndarray.flatten(gray).reshape(1, -1), axis=0)
        i+=1
    BATCH_DATA = np.reshape(BATCH_DATA, BATCH_DATA.shape + (1,))
    BATCH_DATA/=255.0
    NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape

    # file input dataset testing
    test_path = "C:/Users/LENOVO/Downloads/data75test"
    test_dir = os.listdir(test_path)   

    def get_dataset_test(filename):
        frame = cv.imread(test_path+'/'+filename)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return frame, gray

    TEST_BATCH_DATA = np.empty([0, 5625])
    i= 1
    for filename in test_dir:
        print(str(i) + "----- " +filename)
        frame, gray = get_dataset_test(filename)
        TEST_BATCH_DATA = np.append(TEST_BATCH_DATA, np.ndarray.flatten(gray).reshape(1, -1), axis=0)
        i+=1
    TEST_BATCH_DATA = np.reshape(TEST_BATCH_DATA, TEST_BATCH_DATA.shape + (1,))
    TEST_BATCH_DATA/=255.0
    
    W1 = np.append(VISHID, HIDRECBIASES.reshape(1, -1), axis = 0)
    W2 = np.append(HIDPEN, PENRECBIASES.reshape(1, -1), axis = 0)
    W3 = np.append(HIDPEN2, PENRECBIASES2.reshape(1, -1), axis = 0)
    W4 = np.append(HIDPEN3, PENRECBIASES3.reshape(1, -1), axis = 0)
    W5 = np.append(HIDPEN4, PENRECBIASES4.reshape(1, -1), axis = 0)
    W6 = np.append(HIDPEN5, PENRECBIASES5.reshape(1, -1), axis = 0)
    W7 = np.append(HIDPEN6, PENRECBIASES6.reshape(1, -1), axis = 0)
    W8 = np.append(HIDTOP, TOPRECBIASES.reshape(1, -1), axis = 0)
    W9 = np.append(HIDTOP.T, TOPGENBIASES.reshape(1, -1), axis = 0)
    W10 = np.append(HIDPEN6.T, HIDGENBIASES6.reshape(1, -1), axis = 0)
    W11 = np.append(HIDPEN5.T, HIDGENBIASES5.reshape(1, -1), axis = 0)
    W12 = np.append(HIDPEN4.T, HIDGENBIASES4.reshape(1, -1), axis = 0)
    W13 = np.append(HIDPEN3.T, HIDGENBIASES3.reshape(1, -1), axis = 0)
    W14 = np.append(HIDPEN2.T, HIDGENBIASES2.reshape(1, -1), axis = 0)
    W15 = np.append(HIDPEN.T, HIDGENBIASES.reshape(1, -1), axis = 0)
    W16 = np.append(VISHID.T, VISBIASES.reshape(1, -1), axis = 0)

    L1 = W1.shape[0]-1
    L2 = W2.shape[0]-1
    L3 = W3.shape[0]-1
    L4 = W4.shape[0]-1
    L5 = W5.shape[0]-1
    L6 = W6.shape[0]-1
    L7 = W7.shape[0]-1
    L8 = W8.shape[0]-1
    L9 = W9.shape[0]-1
    L10 = W10.shape[0]-1
    L11 = W11.shape[0]-1
    L12 = W12.shape[0]-1
    L13 = W13.shape[0]-1
    L14 = W14.shape[0]-1
    L15 = W15.shape[0]-1
    L16 = W16.shape[0]-1
    L17 = L1

    TEST_ERR=[]
    TRAIN_ERR=[]

    for epoch in range(1, MAX_EPOCH):
        ERR = 0
        NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape
        N = NUM_CASES
        for batch in range(0, NUM_BATCHES):
            data = BATCH_DATA[:,:,batch]
            data = np.append(data, np.ones((N, 1)), axis=1)
            
            W1_PROBS = 1.0/(1 + np.exp(np.matmul(-data, W1)))
            W1_PROBS = np.append(W1_PROBS, np.ones((N, 1)), axis=1)
            W2_PROBS = 1.0/(1 + np.exp(np.matmul(-W1_PROBS, W2)))
            W2_PROBS = np.append(W2_PROBS, np.ones((N, 1)), axis=1)
            W3_PROBS = 1.0/(1 + np.exp(np.matmul(-W2_PROBS,W3)))
            W3_PROBS = np.append(W3_PROBS, np.ones((N, 1)), axis=1)
            W4_PROBS = 1.0/(1 + np.exp(np.matmul(-W3_PROBS,W4)))
            W4_PROBS = np.append(W4_PROBS, np.ones((N, 1)), axis=1)
            W5_PROBS = 1.0/(1 + np.exp(np.matmul(-W4_PROBS,W5)))
            W5_PROBS = np.append(W5_PROBS, np.ones((N, 1)), axis=1)
            W6_PROBS = 1.0/(1 + np.exp(np.matmul(-W5_PROBS,W6)))
            W6_PROBS = np.append(W6_PROBS, np.ones((N, 1)), axis=1)
            W7_PROBS = 1.0/(1 + np.exp(np.matmul(-W6_PROBS,W7)))
            W7_PROBS = np.append(W7_PROBS, np.ones((N, 1)), axis=1)

            W8_PROBS_TRAIN = np.matmul(W7_PROBS, W8)
            W8_PROBS = np.append(W8_PROBS_TRAIN, np.ones((N, 1)), axis=1)

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
            DATAOUT = 1.0/(1 + np.exp(np.matmul(-W15_PROBS, W16)))
            ERR += 1/N*np.sum(np.sum(np.square(data[:,:-1]-DATAOUT), axis=0), axis=0)
        TRAIN_ERR = ERR/NUM_BATCHES

        OUTPUT = np.array([])
        for ii in range(15):
            A = np.append(data[ii, :-1].T, DATAOUT[ii, :].T)
            A = A.reshape(5625, 2)
            OUTPUT = np.append(OUTPUT, A)

        TESTNUMCASES, TESTNUMDIMS, TESTNUMBATCHES = TEST_BATCH_DATA.shape
        N = TESTNUMCASES
        ERR = 0
        for batch in range(0, TESTNUMBATCHES):
            data = TEST_BATCH_DATA[:,:,batch]
            data = np.append(data, np.ones((N, 1)), axis=1)

            W1_PROBS = 1.0/(1 + np.exp(np.matmul(-data, W1)))
            W1_PROBS = np.append(W1_PROBS, np.ones((N, 1)), axis=1)
            W2_PROBS = 1.0/(1 + np.exp(np.matmul(-W1_PROBS, W2)))
            W2_PROBS = np.append(W2_PROBS, np.ones((N, 1)), axis=1)
            W3_PROBS = 1.0/(1 + np.exp(np.matmul(-W2_PROBS,W3)))
            W3_PROBS = np.append(W3_PROBS, np.ones((N, 1)), axis=1)
            W4_PROBS = 1.0/(1 + np.exp(np.matmul(-W3_PROBS,W4)))
            W4_PROBS = np.append(W4_PROBS, np.ones((N, 1)), axis=1)
            W5_PROBS = 1.0/(1 + np.exp(np.matmul(-W4_PROBS,W5)))
            W5_PROBS = np.append(W5_PROBS, np.ones((N, 1)), axis=1)
            W6_PROBS = 1.0/(1 + np.exp(np.matmul(-W5_PROBS,W6)))
            W6_PROBS = np.append(W6_PROBS, np.ones((N, 1)), axis=1)
            W7_PROBS = 1.0/(1 + np.exp(np.matmul(-W6_PROBS,W7)))
            W7_PROBS = np.append(W7_PROBS, np.ones((N, 1)), axis=1)

            W8_PROBS_TEST = np.matmul(W7_PROBS, W8)
            W8_PROBS = np.append(W8_PROBS_TEST, np.ones((N, 1)), axis=1)

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
            DATAOUT = 1.0/(1 + np.exp(np.matmul(-W15_PROBS, W16)))
            ERR += 1/N*np.sum(np.sum(np.square(data[:,:-1]-DATAOUT), axis=0), axis=0)
        TEST_ERR = ERR/NUM_BATCHES
        print('Before epoch {} Train squared error: {} Test squared error: {}'.format(epoch,TRAIN_ERR,TEST_ERR))

        TT = 0
        TT+=1
        data=np.empty((81,5625), int)
            
        MAX_ITER = 3
        VV = np.concatenate((W1.reshape(1,-1), W2.reshape(1,-1), W3.reshape(1,-1),
                W4.reshape(1,-1), W5.reshape(1,-1), W6.reshape(1, -1), W7.reshape(1, -1),
                W8.reshape(1, -1), W9.reshape(1,-1), W10.reshape(1,-1), W11.reshape(1,-1),
                W12.reshape(1,-1), W13.reshape(1,-1), W14.reshape(1, -1), W15.reshape(1, -1),
                W16.reshape(1, -1)), axis=1)
        DIM = np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13, L14, L15, L16, L17]).reshape(1, -1).T

        f, df = CG_MNIST(VV, DIM, data)
        X, fX, i = minimize(VV, f, df, MAX_ITER, DIM, data)

        W1 = X[0][0:(L1+1)*L2].reshape(L1+1,L2)
        X3 = (L1+1)*L2
        W2 = X[0][X3:X3+(L2+1)*L3].reshape(L2+1, L3)
        X3 = X3+(L2+1)*L3
        W3 = X[0][X3:X3+(L3+1)*L4].reshape(L3+1, L4)
        X3 = X3+(L3+1)*L4
        W4 = X[0][X3:X3+(L4+1)*L5].reshape(L4+1, L5)
        X3 = X3+(L4+1)*L5
        W5 = X[0][X3:X3+(L5+1)*L6].reshape(L5+1, L6)
        X3 = X3+(L5+1)*L6
        W6 = X[0][X3:X3+(L6+1)*L7].reshape(L6+1, L7)
        X3 = X3+(L6+1)*L7
        W7 = X[0][X3:X3+(L7+1)*L8].reshape(L7+1, L8)
        X3 = X3+(L7+1)*L8
        W8 = X[0][X3:X3+(L8+1)*L9].reshape(L8+1, L9)

        X3 = X3+(L8+1)*L9
        W9 = X[0][X3:X3+(L9+1)*L10].reshape(L9+1, L10)
        X3 = X3+(L9+1)*L10
        W10 = X[0][X3:X3+(L10+1)*L11].reshape(L10+1, L11)
        X3 = X3+(L10+1)*L11
        W11 = X[0][X3:X3+(L11+1)*L12].reshape(L11+1, L12)
        X3 = X3+(L11+1)*L12
        W12 = X[0][X3:X3+(L12+1)*L13].reshape(L12+1, L13)
        X3 = X3+(L12+1)*L13
        W13 = X[0][X3:X3+(L13+1)*L14].reshape(L13+1, L14)
        X3 = X3+(L13+1)*L14
        W14 = X[0][X3:X3+(L14+1)*L15].reshape(L14+1, L15)
        X3 = X3+(L14+1)*L15
        W15 = X[0][X3:X3+(L15+1)*L16].reshape(L15+1, L16)
        X3 = X3+(L15+1)*L16
        W16 = X[0][X3:X3+(L16+1)*L17].reshape(L16+1, L17)
    return ERR
