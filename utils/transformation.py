import numpy as np

import random
from scipy.spatial.transform import Rotation as R

def rotation_2_quaternion(rot):
    r = R.from_matrix(rot)
    q_array = R.as_quat(r)
    return q_array # retrun numpy array

def rotateVecRot(vec, rot):
    r = R.from_matrix(rot)
    return r.apply(vec)


def rotateVecQuat(vec, quat):
    r = R.from_quat(quat)
    return r.apply(vec)


def quaternion_2_rotation(q):
    n = np.dot(q, q)
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-(q[1, 1]+q[2, 2]), -(q[2, 3]-q[1, 0]), (q[1, 3]+q[2, 0])],
        [q[2, 3]+q[1, 0], -(1.0-(q[1, 1]+q[3, 3])), (q[1, 2]-q[3, 0])],
        [-(q[1, 3]-q[2, 0]), (q[1, 2]+q[3, 0]), -(1.0-(q[2, 2]+q[3, 3]))]])


def quaternion_2_matrix(quat):
    q = np.array(quat[3:7],dtype=np.float64, copy=True)
    n = np.dot(q, q)
    # epsilon for testing whether a number is close to zero
    _EPS = np.finfo(float).eps * 4.0
    if n < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-(q[1, 1]+q[2, 2]), -(q[2, 3]-q[1, 0]), (q[1, 3]+q[2, 0]), quat[0]],
        [q[2, 3]+q[1, 0], -(1.0-(q[1, 1]+q[3, 3])), (q[1, 2]-q[3, 0]), quat[1]],
        [-(q[1, 3]-q[2, 0]), (q[1, 2]+q[3, 0]), -(1.0-(q[2, 2]+q[3, 3])), quat[2]],
        [0.0, 0.0, 0.0, 1.0]])


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-8

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    # assert(isRotationMatrix(R))
     
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-8
 
    if  not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
    
def eulerAnglesToRotationMatrix(theta):
     
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0, np.sin(theta[0]), np.cos(theta[0])  ]
                    ])


   
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0], 
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])
        
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
            
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


def randamPose(translation):
    randomAngles = [0, 0, 0]
    randomAngles[2] = random.random()*math.pi*2.0
    transformMatrix = np.eye(4)
    transformMatrix[:3,:3] = eulerAnglesToRotationMatrix(randomAngles)
    transformMatrix[:3,-1] = np.array(translation)
    
    return transformMatrix
    

def refineRotationTransform(T):
    return_T = T     
    R = T[:3,:3] 
    angles = rotationMatrixToEulerAngles(R)
    angles[0] = 0
    angles[1] = 0
    
    R_new = eulerAnglesToRotationMatrix(angles)
    return_T[:3,:3] = R_new
    return return_T


def lookat2rotation(vec_x, vec_y, vec_z):
    vec_x = vec_x.reshape((1, 3))
    vec_y = vec_y.reshape((1, 3))
    vec_z = vec_z.reshape((1, 3))

    vec_x = vec_x / np.linalg.norm(vec_x)
    vec_y = vec_y / np.linalg.norm(vec_y)
    vec_z = vec_z / np.linalg.norm(vec_z)

    rot = np.transpose(np.vstack((vec_x, vec_y, vec_z)))

    return rot

## Verified
def lookat2RotationTransform(v,t,option = "RightHand"):
    # ref link: https://www.scichart.com/documentation/win/current/Orientation%20(3D%20Space)%20in%20the%20SciChart3DSurface.html
    T = np.eye(4)
    if option == "RightHand":# Zup
        world_up = np.array([0,0,1]).reshape((1,3)) # all in 3 by 1 vector
        if v.shape[0] == 1:
            cam_forward = v
        else:
            cam_forward = np.transpose(v)
        if np.sum(np.cross(cam_forward, world_up)) == 0:
            world_up = np.array([0,1,1]).reshape((1,3))
            
        cam_right = np.cross(cam_forward, world_up)
        cam_right= cam_right/np.linalg.norm(cam_right)
        
        cam_up= np.cross(cam_right, cam_forward)
        cam_up = cam_up/np.linalg.norm(cam_up)
#        ### right hand trial
        R = np.transpose(np.vstack((cam_right, -cam_up, cam_forward))) #ok
        T[:3,:3] = R
        T[:3,3] = np.transpose(t)
    else:
        print("Not implemented yet")
#        R = np.transpose(np.vstack((-cam_up, -cam_right, cam_forward))) # bad
#        R = np.transpose(np.vstack((-cam_right, cam_up, cam_forward))) # bad
#        R = np.transpose(np.vstack((cam_up, cam_right, cam_forward))) #bad
         ### left hand trial (Not left handed!)
#        R = np.transpose(np.vstack((cam_right, cam_up, cam_forward))) #ok verified, better than right hand trial, but seems up side down
#        R = np.transpose(np.vstack((cam_up, -cam_right, cam_forward))) #bad 
#        R = np.transpose(np.vstack((-cam_right, -cam_up, cam_forward)))#bad
#        R = np.transpose(np.vstack((-cam_up, cam_right, cam_forward)))#bad
        
        ## other trial
#        R = np.transpose(np.vstack((cam_right, cam_up, -cam_forward))) #bad
#        R = np.transpose(np.vstack((cam_right, -cam_up,-cam_forward))) 
        
        
    
    return T
    
    
    
    
    