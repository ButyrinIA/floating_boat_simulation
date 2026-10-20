def Rot3D(x,y,z, q0,q1,q2,q3):
    RX = x * (q0**2 + q1**2 - q2**2 - q3**2) + y * (- 2*q0*q3 + 2*q1*q2) + z * (2*q1*q3 + 2*q0*q2)
    RY = x * (2*q0*q3 + 2*q1*q2) + y * (q0**2 - q1**2 + q2**2 - q3**2) + z * (- 2*q0*q1 + 2*q2*q3)
    RZ = x * (2*q1*q3 - 2*q0*q2) + y * (2*q0*q1 + 2*q2*q3) + z * (q0**2 - q1**2 - q2**2 + q3**2)
    return RX, RY, RZ

def TRot3D(X,Y,Z, q0,q1,q2,q3):
    Rx = X * (q0**2 + q1**2 - q2**2 - q3**2) + Y * (2*q0*q3 + 2*q1*q2) + Z * (2*q1*q3 - 2*q0*q2)
    Ry = X * (- 2*q0*q3 + 2*q1*q2) + Y * (q0**2 - q1**2 + q2**2 - q3**2) + Z * (2*q0*q1 + 2*q2*q3)
    Rz = X * (2*q1*q3 + 2*q0*q2) + Y * (- 2*q0*q1 + 2*q2*q3) + Z * (q0**2 - q1**2 - q2**2 + q3**2)
    return Rx, Ry, Rz

