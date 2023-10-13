import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    stack_ = []
    offset_tmp = []

    with open(bvh_file_path, 'r') as f:
        line = f.readline().strip()  # First line HIERARCHY
        while f.readable():
            line = f.readline().strip()
            if line.startswith("MOTION"):
                break
            # 1. KeyWord, Root/Joint
            #    Read the next line, suppose to be {
            #    压栈
            if line.upper().startswith("ROOT"):
                jname = line.split()[1]
                joint_name.append(jname)
                joint_parent.append(-1)
                stack_.append((jname, len(joint_parent) - 1))
                line = f.readline()
                continue

            if line.upper().startswith("JOINT"):
                jname = line.split()[1]
                joint_name.append(jname)
                parentTuple = stack_[len(stack_) - 1]
                joint_parent.append(parentTuple[1])
                stack_.append((jname, len(joint_parent) - 1))
                line = f.readline()
                continue

            # 2. OFFSET, CHANNELS
            #
            if line.upper().startswith("OFFSET"):
                line = line[6:]
                data = [float(x) for x in line.split()]
                offset_tmp.append(data)
                continue

            if line.upper().startswith("CHANNELS"):
                continue

            # 3. End Site
            #    name is parent name + '_end'
            if line.startswith("End Site"):
                parent = stack_[len(stack_) - 1]
                myName = parent[0] + "_end"
                joint_name.append(myName)
                parentId = parent[1]
                joint_parent.append(parentId)

                f.readline()  # skip '{'
                line = f.readline().strip()
                line = line[6:]
                data = [float(x) for x in line.split()]
                offset_tmp.append(data)
                f.readline()
                continue

            # 4. }
            # 弹栈
            if line.startswith("}"):
                stack_.pop()
                line = ''
                continue

        pass
    joint_offset = np.array(offset_tmp).reshape(-1, 3)
    return joint_name, joint_parent, joint_offset



def part2_forward_kinematics2(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
      输入: part1 获得的关节名字，父节点列表，偏移量列表
          motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
          frame_id: int，需要返回的帧的索引
      输出:
          joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
          joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
      Tips:
          1. joint_orientations的四元数顺序为(x, y, z, w)
          2. from_euler时注意使用大写的XYZ
      """
    ###### 用欧拉角实现
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))
    buffer = motion_data[frame_id]
    idx_index = 0

    for idx, offset in enumerate(joint_offset):
        curr_joint_name = joint_name[idx]
        parent_index = joint_parent[idx]
        # print("%d, %s, %d"%(idx, joint_name[idx], idx_index))
        if curr_joint_name.startswith("RootJoint"):
            joint_positions[idx] = buffer[:3]
            joint_orientations[idx] = R.from_euler('XYZ', buffer[3:6], degrees=True).as_quat()
            idx_index += 6
            pass

        elif curr_joint_name.endswith("_end"):
            po = R.from_quat(joint_orientations[parent_index]).as_matrix()
            joint_positions[idx] = joint_positions[parent_index] + np.dot(po, offset)
            pass

        else:
            rotation = R.from_euler('XYZ', buffer[idx_index: idx_index + 3], degrees=True)
            rotation_p = R.from_quat(joint_orientations[parent_index])
            tmp = (rotation_p * rotation).as_quat()
            joint_orientations[idx] = tmp

            joint_positions[idx] = np.dot(R.from_quat(joint_orientations[parent_index]).as_matrix(), offset) + joint_positions[parent_index]
            idx_index += 3
        pass

    return joint_positions, joint_orientations



def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))
    buffer = motion_data[frame_id]
    idx_index = 0
    for idx, offset in enumerate(joint_offset):
        curr_joint_name = joint_name[idx]
        parent_index = joint_parent[idx]
        #print("%d, %s, %d"%(idx, joint_name[idx], idx_index))
        if curr_joint_name.startswith("RootJoint"):
            joint_positions[idx] = buffer[:3]
            joint_orientations[idx] = R.from_euler('XYZ', buffer[3:6], degrees=True).as_quat()
            idx_index += 6
            pass

        elif curr_joint_name.endswith("_end"):
            po = R.from_quat(joint_orientations[parent_index]).as_matrix()
            q_result = po.dot(offset)
            joint_positions[idx] = joint_positions[parent_index] + q_result
            #idx_index +=3
            pass

        else:
            rotation = R.from_euler('XYZ', buffer[idx_index: idx_index+3], degrees=True).as_matrix()
            rotation_p = R.from_quat(joint_orientations[parent_index]).as_matrix()
            tmp = rotation_p.dot(rotation)
            joint_orientations[idx] = R.from_matrix(tmp).as_quat()
            joint_positions[idx] = rotation_p.dot(offset) + joint_positions[parent_index]
            idx_index +=3
            pass
        pass

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name_t, joint_parent_t, joint_offset_t = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a, joint_parent_a, joint_offset_a = part1_calculate_T_pose(A_pose_bvh_path)

    joint_name_t_ankle = []
    joint_name_a_ankle = []
    map_aTot = {}
    Q_aTot = np.zeros((len(joint_parent_a), 4))



    for idx, jname in enumerate(joint_name_t):
        if jname.endswith('_end'):
            continue
        joint_name_t_ankle.append(jname)

    for idx, jname in enumerate(joint_name_a):
        if jname.endswith('_end'):
            continue
        joint_name_a_ankle.append(jname)

    for idx, jname in enumerate(joint_name_t_ankle):
        map_aTot[jname] = idx



    motion_data = load_motion_data(A_pose_bvh_path)
    for i in range(len(motion_data)):
        buffer = motion_data[i][3:]
        for j, jname in enumerate(joint_name_a_ankle):
            if jname.startswith('RootJoint'):
                continue
            if map_aTot[jname] == j:
                continue

            jj = map_aTot[jname]
            tmp = buffer[j*3:j*3+3]
            buffer[j*3:j*3+3] = buffer[jj*3:jj*3+3]
            buffer[jj*3:jj*3+3] = tmp

        pass
    return motion_data


def part3_retarget_func2(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出:
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    def index_bone_to_channel(index, flag):
        if flag == 't':
            end_bone_index = end_bone_index_t
        else:
            end_bone_index = end_bone_index_a
        for i in range(len(end_bone_index)):
            if end_bone_index[i] > index:
                return index - i
        return index - len(end_bone_index)

    def get_t2a_offset(bone_name):
        l_bone = ['lShoulder', 'lElbow', 'lWrist']
        r_bone = ['rShoulder', 'rElbow', 'rWrist']
        if bone_name in l_bone:
            return R.from_euler('XYZ', [0., 0., 45.], degrees=True)
        if bone_name in r_bone:
            return R.from_euler('XYZ', [0., 0., -45.], degrees=True)
        return R.from_euler('XYZ', [0., 0., 0.], degrees=True)

    motion_data = load_motion_data(A_pose_bvh_path)

    t_name, t_parent, t_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_name, a_parent, a_offset = part1_calculate_T_pose(A_pose_bvh_path)

    end_bone_index_t = []
    for i in range(len(t_name)):
        if t_name[i].endswith('_end'):
            end_bone_index_t.append(i)

    end_bone_index_a = []
    for i in range(len(a_name)):
        if a_name[i].endswith('_end'):
            end_bone_index_a.append(i)

    for m_i in range(len(motion_data)):
        frame = motion_data[m_i]
        cur_frame = np.empty(frame.shape[0])
        cur_frame[:3] = frame[:3]
        for t_i in range(len(t_name)):
            cur_bone = t_name[t_i]
            a_i = a_name.index(t_name[t_i])
            if cur_bone.endswith('_end'):
                continue
            channel_t_i = index_bone_to_channel(t_i, 't')
            channel_a_i = index_bone_to_channel(a_i, 'a')

            # retarget
            local_rotation = frame[3 + channel_a_i * 3: 6 + channel_a_i * 3]
            if cur_bone in ['lShoulder', 'lElbow', 'lWrist', 'rShoulder', 'rElbow', 'rWrist']:
                p_bone_name = t_name[t_parent[t_i]]
                Q_pi = get_t2a_offset(p_bone_name)
                Q_i = get_t2a_offset(cur_bone)
                local_rotation = (Q_pi * R.from_euler('XYZ', local_rotation, degrees=True) * Q_i.inv()).as_euler('XYZ',
                                                                                                                 degrees=True)
            cur_frame[3 + channel_t_i * 3: 6 + channel_t_i * 3] = local_rotation

        motion_data[m_i] = cur_frame

    return motion_data