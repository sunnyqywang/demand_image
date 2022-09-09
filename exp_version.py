def get_hp_from_version_code(v1, v2):
    if v1 is None:
        l = None
    elif v1 == 'A':
        l = 1e-4
    elif v1 == 'B':
        l = 1e-3
    elif v1 == 'C':
        l = 1e-1
    elif v1 == 'D':
        l = 1
    elif v1 == 'E':
        l = 1e+1
    elif v1 == 'F':
        l = 1e+3
    elif v1 == 'G':
        l = 1e+2
    elif v1 == 'H':
        l = 1e+4
    elif v1 == 'I':
        l = 1e-2
    else:
        l = v1
    
#         raise Exception("Lambda version code error")
        
    if v2 == 1:
        lr = 1e-3
        wd = 1e-4
    elif v2 == 2:
        lr = 2e-3
        wd = 1e-4
    elif v2 == 3:
        lr = 1e-3
        wd = 1e-3
    elif v2 == 5:
        lr = 5e-4
        wd = 1e-4
    else:
        raise Exception("Lr Wd version code error")
    
    return l, lr, wd
