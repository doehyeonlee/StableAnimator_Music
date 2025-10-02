import h5py

filename = "/root/dataset/rec/gJS_sFM_cAll_d02_mJS5_ch10.h5"
with h5py.File(filename, "r") as f:
    # 최상위 키들 보기
    print("Keys:", list(f.keys()))

    # 특정 데이터셋 shape, dtype 확인
    for k in f.keys():
        dset = f[k]
        print(f"{k}: shape={dset.shape}, dtype={dset.dtype}")