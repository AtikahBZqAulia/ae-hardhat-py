def rbmhidlinear(BATCH_DATA, RESTART, NUM_HID, NUM_DIMS, MAX_EPOCH):
    import scipy.io as sio
    import numpy as np
    import matplotlib.pyplot as plt

    EPSILON_W = 0.001
    EPSILON_VB = 0.001
    EPSILON_HB = 0.001
    WEIGHT_COST = 0.0002
    INITIAL_MOMENTUM = 0.5
    FINAL_MOMENTUM = 0.9

    print(BATCH_DATA.shape)
    NUM_CASES, NUM_DIMS, NUM_BATCHES = BATCH_DATA.shape

    if RESTART==1:
        RESTART = 0
        EPOCH = 0

        VISHID = 0.1 * np.random.randn(NUM_DIMS, NUM_HID)
        HIDBIASES = np.zeros(NUM_HID)
        VISBIASES = np.zeros(NUM_DIMS)
        POSHIDPROBS = np.zeros((NUM_CASES, NUM_HID))
        NEGHIDPROBS = np.zeros((NUM_CASES, NUM_HID))
        POSPRODS = np.zeros((NUM_DIMS, NUM_HID))
        NEGPRODS = np.zeros((NUM_DIMS, NUM_HID))
        VISHIDINC = np.zeros((NUM_DIMS, NUM_HID))
        HIDBIASINC = np.zeros(NUM_HID)
        VISBIASINC = np.zeros(NUM_DIMS)
        BATCHPOSHIDPROBS = np.zeros((NUM_CASES, NUM_HID, NUM_BATCHES))
    LST_ERR = []
    for epoch in range(EPOCH, MAX_EPOCH):
        print('epoch {}'.format(epoch))
        ERR_SUM = 0
        for batch in range(NUM_BATCHES):
            print('epoch {} batch {}'.format(epoch,batch))
            
            data = BATCH_DATA[:,:,batch]
            MULT  = np.matmul(-data,VISHID)
            KURANG = np.tile(HIDBIASES, (NUM_CASES, 1))
            POSHIDPROBS = 1.0/(1 + (np.exp(MULT- KURANG)))
            BATCHPOSHIDPROBS[:,:,batch] = POSHIDPROBS
            POSPRODS = np.matmul(data.T,POSHIDPROBS)
            POSHIDACT = np.sum(POSHIDPROBS, axis=0)
            POSVISACT = np.sum(data, axis=0)

            POSHIDSTATES = POSHIDPROBS + np.random.randn(NUM_CASES, NUM_HID)
            NEGDATA = 1.0/(1 + np.exp(np.matmul(-POSHIDSTATES, VISHID.T) - np.tile(VISBIASES, (NUM_CASES, 1))))
            NEGHIDPROBS = 1.0/(1 + np.exp(np.matmul(-NEGDATA, VISHID) - np.tile(HIDBIASES, (NUM_CASES,1))))
            NEGPRODS = np.matmul(NEGDATA.T, NEGHIDPROBS)
            NEGHIDACT = np.sum(NEGHIDPROBS, axis=0)
            NEGVISACT = np.sum(NEGDATA, axis=0)

            ERR = np.sum(np.sum(np.square(data-NEGDATA), axis=0), axis=0)
            ERR_SUM += ERR

            if epoch>5:
                MOMENTUM = FINAL_MOMENTUM
            else:
                MOMENTUM = INITIAL_MOMENTUM

            VISHIDINC = MOMENTUM* VISHIDINC + EPSILON_W  * ((POSPRODS-NEGPRODS)/NUM_CASES - WEIGHT_COST*VISHID)
            VISBIASINC = MOMENTUM* VISBIASINC + (EPSILON_VB/NUM_CASES) *(POSVISACT-NEGVISACT)
            HIDBIASINC = MOMENTUM * HIDBIASINC + (EPSILON_HB/NUM_CASES)* (POSHIDACT-NEGHIDACT)
            VISHID += VISHIDINC
            VISBIASES += VISBIASINC
            HIDBIASES += HIDBIASINC
        print('epoch {} error {}'.format(epoch, ERR_SUM))
        LST_ERR.append(ERR_SUM)
    plt.plot(LST_ERR)
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Epoch')
    plt.show()
    return VISHID, HIDBIASES, VISBIASES

