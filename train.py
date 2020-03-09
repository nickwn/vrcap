import numpy as np
from keras.models import Sequential
from keras import layers
from pyfbx import parse_bin

class Curve:
    def eval(self, t):
        return 0

# cubic hermite spline
class CubicCurve(Curve):
    def __init__(self, p1, p2, t1, t2):
        self.p1 = p1
        self.p2 = p2
        self.t1 = t1
        self.t2 = t2

    def _h00(self, t):
        return 2 * (t ** 3) - 3 * (t ** 2) + 1

    def _h10(self, t):
        return (t ** 3) - 2 * (t ** 2) + t

    def _h01(self, t):
        return -2 * (t ** 3) + 3 * (t ** 2)
    
    def _h11(self, t):
        return (t ** 3) - (t ** 2)

    def eval(self, t):
        return _h00(t) * self.p1 + _h10(t) * (self.p1 + self.t1) + _h01(t) * (self.p2 - self.t2) + _h11(t) * self.p2

# standard linear interpolation
class LinearCurve(Curve):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def eval(self, t):
        return self.p1 + (self.p2 - self.p1) * t

class CurveSet:
    def __init__(self, times, curves): # where len(times) = len(curves) + 1
        self.curves = curves
        self.times = times

    def eval(self, t):
        high_t_idx = 0
        while(self.times[high_t_idx] <= t):
            high_t_idx += 1
        
        low_t_idx = high_t_idx - 1
        norm_t = (t - self.times[low_t_idx]) / (self.times[high_t_idx] - self.times[low_t_idx])

        return self.curves[low_t_idx].eval(norm_t)

    def get_times(self):
        return self.times

def find_id(id, elems):
    for e in elems:
        if e.id == id:
            return e

def find_id_all(id, elems):
    all_found = []
    for e in elems:
        if e.id == id:
            all_found.append(e)
    return all_found

def make_name_to_ids(elems):
    name_to_ids = {}
    for e in elems:
        name_to_ids[e.props[1]] = e.props[0]
    return name_to_ids

def make_parent_to_child(elems):
    p_to_c = {}
    for e in elems:
        if not e.props[2] in p_to_c:
            p_to_c[e.props[2]] = []
        connection_info = {'id': e.props[1]}
        if len(e.props) > 3:
            connection_info['type'] = e.props[3]
        p_to_c[e.props[2]].append(connection_info)
    return p_to_c

def make_transform_anim_dict(model_name, name_to_ids, ids_to_anims, parent_to_child, verbose=False):
    transform_anim = {}

    component_types = ['Lcl Translation', 'Lcl Rotation', 'Lcl Scaling']
    dim_types = ['d|X', 'd|Y', 'd|Z']

    connection_stack = [] # has just id numbers
    curr_component = '' # either Lcl Translation or Lcl Rotation
    curr_dim = '' # either d|X d|Y d|Z

    full_model_name = next(k for k in name_to_ids.keys() if model_name in k)
    root_id = name_to_ids[full_model_name]
    connection_stack.append({'id': root_id, 'is_leaf': False})

    while len(connection_stack) > 0:
        curr_node = connection_stack.pop()
        is_leaf = False

        if 'type' in curr_node:
            if curr_node['type'] in component_types:
                curr_component = curr_node['type']
            elif curr_node['type'] in dim_types:
                curr_dim = curr_node['type']
                is_leaf = True

        if is_leaf:
            if not curr_component in transform_anim:
                transform_anim[curr_component] = {}
            leaf_node = curr_node['id']
            if not leaf_node in ids_to_anims:
                if verbose:
                    print('curve for model %s, component %s, dim %s not found' % (model_name, curr_component, curr_dim))
            transform_anim[curr_component][curr_dim] = leaf_node
        else:
            if curr_node['id'] == 0:
                if verbose:
                    print('hit root')
            elif not curr_node['id'] in parent_to_child:
                if verbose:
                    print('key %d not found, skipping' % curr_node['id'])
            else:
                for child in parent_to_child[curr_node['id']]:
                    connection_stack.append(child)
    
    return transform_anim
        
def make_id_to_anim(anim_curves):
    id_to_anim = {}
    for anim_curve in anim_curves:
        anim_id = anim_curve.props[0]
        key_times = find_id('KeyTime', anim_curve.elems).props[0]
        key_vals = find_id('KeyValueFloat', anim_curve.elems).props[0]

        curves = []
        for i in range(len(key_vals)-1):
            curves.append(LinearCurve(key_vals[i], key_vals[i + 1]))
        
        id_to_anim[anim_id] = CurveSet(key_times, curves)

    return id_to_anim

def get_max_time_val(id_to_anims):
    max_time = 0
    for key, curve_set in id_to_anims.items():
        curr_max_time = curve_set.get_times()[-1]
        max_time = max(max_time, curr_max_time)
    
    return max_time

anims_to_use = ['anims/Walk_06_Look_Around_Loop2.fbx', 'anims/Convo_01_Low_Key_Loop.fbx', 'anims/Convo_11_Listening_Loop.fbx',
    'anims/MOB1_Jog_F.fbx', 'anims/MOB1_Run_F.fbx', 'anims/MOB1_Stand_Relaxed_Fgt_v4.fbx', 'anims/MOB1_Stand_Relaxed_Idle_v2.fbx',
    'anims/MOB1_Walk_F.fbx', 'anims/SCR_Walk_Scared_Fwd_Look_Left_Loop.fbx', 'anims/SCR_Walk_Scared_Fwd_Look_Right_Loop.fbx',
    'anims/Walk_02_Cheerful_Loop.fbx', 'anims/Walk_04_Texting_Loop.fbx', 'anims/Walk_08_Listen_Music_Loop.fbx', 
    'anims/Walk_13_Power_Walk_Loop.fbx']

extra_anims = ['anims/extras/Crouch_Idle_Rifle_Hip.fbx', 'anims/extras/Idle_Rifle_Hip.fbx', 'anims/extras/Prone_Idle.fbx', 
    'anims/extras/Crouch_Idle_Rifle_Ironsights.fbx', 'anims/extras/Idle_Rifle_Hip_Break1.fbx', 'anims/extras/Prone_Reload_Rifle.fbx', 
    'anims/extras/Crouch_to_Stand_Rifle_Hip.fbx', 'anims/extras/Idle_Rifle_Hip_Break2.fbx', 'anims/extras/Prone_Reload_Shotgun.fbx', 
    'anims/extras/Death_1.fbx', 'anims/extras/Idle_Rifle_Ironsights.fbx', 'anims/extras/Prone_To_Stand.fbx',
    'anims/extras/Death_2.fbx', 'anims/extras/MOB1_CrouchWalk_F.fbx', 'anims/extras/Reload_Pistol.fbx', 'anims/extras/Death_3.fbx', 
    'anims/extras/MOB1_Crouch_Idle_V2.fbx', 'anims/extras/Reload_Rifle_Hip.fbx', 'anims/extras/Equip_Pistol_Standing.fbx', 
    'anims/extras/MOB1_Stand_Relaxed_Death_B.fbx', 'anims/extras/Stand_To_Prone.fbx', 'anims/extras/Equip_Rifle_Standing.fbx',
    'anims/extras/Prone_Death_1.fbx', 'anims/extras/Stand_to_Crouch_Rifle_Hip.fbx', 'anims/extras/Idle_Pistol.fbx', 
    'anims/extras/Prone_Death_2.fbx']

#anims_to_use = extra_anims
for extra_anim in extra_anims:
    anims_to_use.append(extra_anim)
#anims_to_use = ['anims/Walk_06_Look_Around_Loop2.fbx', 'anims/Convo_01_Low_Key_Loop.fbx']

SUBANIM_LEN = 20

all_anim_data = {}

for anim_fn in anims_to_use:
    print('parsing %s...' % anim_fn)

    try:
        fbx_root, fbx_ver = parse_bin.parse(anim_fn)
    except:
        print('error parsing, skipping')
        continue

    objects = find_id('Objects', fbx_root.elems)
    connections = find_id('Connections', fbx_root.elems)

    models = find_id_all('Model', objects.elems)
    anim_curves = find_id_all('AnimationCurve', objects.elems)
    cs = find_id_all('C', connections.elems)

    name_to_ids = make_name_to_ids(models)
    parent_to_child = make_parent_to_child(cs)
    ids_to_anims = make_id_to_anim(anim_curves)

    inputs_to_track = ['root', 'head', 'hand_l', 'hand_r']

    outputs_to_track = ['root', 'head', 'pelvis', 'spine_01', 'spine_02', 'spine_03', 
        'foot_l', 'foot_r', 'calf_l', 'calf_r', 'calf_twist_01_l', 'calf_twist_01_r', 'thigh_l', 'thigh_r',
        'upperarm_l', 'upperarm_r', 'upperarm_twist_01_l', 'upperarm_twist_01_r', 'lowerarm_l', 'lowerarm_r', 
        'hand_l', 'hand_r' ]

    all_items_to_track = inputs_to_track + list(set(outputs_to_track) - set(inputs_to_track)) # combine all w/o duplicates

    anim_transform_dicts = {}
    for item in all_items_to_track:
        anim_transform_dicts[item] = make_transform_anim_dict(item, name_to_ids, ids_to_anims, parent_to_child)

    max_time = get_max_time_val(ids_to_anims)
    print('anim_time: %d' % max_time)
    per_frame = 1000000000 # number of fbx long values per frame

    print('evaluating...')

    # eval for all frames (357 in the case of Walk_06_Look_Around_Loop2.fbx)
    pose_evals = []
    for frame_num in np.arange(0, max_time, per_frame):
        frame_eval = {}
        for body_part, transform_dict in anim_transform_dicts.items():
            body_part_eval = {}
            for component_name, component_curves in transform_dict.items():
                component_eval = {}
                if component_name != 'Lcl Translation' and component_name != 'Lcl Rotation':
                    continue
                for dim_name, dim_curve_id in component_curves.items():
                    if dim_curve_id in ids_to_anims:
                        component_eval[dim_name] = ids_to_anims[dim_curve_id].eval(frame_num)
                    else:
                        component_eval[dim_name] = 0.0 # default eval to zero
                body_part_eval[component_name] = component_eval
            frame_eval[body_part] = body_part_eval
        pose_evals.append(frame_eval)

    if len(pose_evals) > SUBANIM_LEN:
        all_anim_data[anim_fn] = pose_evals
        
# organize into x and y training sets, with shape (NUM_SUBANIMS, SUBANIM_LEN, num float vals)

print('prepping nn...')

attrib_ordering = {
    'Lcl Translation': {
        'd|X': 0,
        'd|Y': 1,
        'd|Z': 2
    },
    'Lcl Rotation': {
        'd|X': 3,
        'd|Y': 4,
        'd|Z': 5
    }
}


num_subanims = 0
for fn, anim_data in all_anim_data.items():
    num_subanims += int(len(anim_data) / SUBANIM_LEN)

TRANSF_FLOATS = 3 * 3 # 3 for translation, 3 for rotation
INPUT_ATTRIBS = TRANSF_FLOATS * len(inputs_to_track)
OUTPUT_ATTRIBS = TRANSF_FLOATS * len(outputs_to_track)

x = np.zeros((num_subanims, SUBANIM_LEN, INPUT_ATTRIBS))
y = np.zeros((num_subanims, SUBANIM_LEN, OUTPUT_ATTRIBS))

subanim_offset = 0
for fn, pose_evals in all_anim_data.items():
    num_splits = int(len(pose_evals) / SUBANIM_LEN)
    subanim_idx = 0
    for i in range(num_splits * SUBANIM_LEN):
        subanim_idx = int(i / (num_splits*SUBANIM_LEN)) + subanim_offset #?
        subanim_pos = i % SUBANIM_LEN
        transf_offset = 0
        for short_body_name in inputs_to_track:
            body_part_name = next(k for k in pose_evals[i].keys() if short_body_name in k)
            for component_name, component_val in pose_evals[i][body_part_name].items():
                for dim_name, dim_val in component_val.items():
                    attrib_idx = attrib_ordering[component_name][dim_name]
                    x[subanim_idx][subanim_pos][attrib_idx + transf_offset] = dim_val
            transf_offset += TRANSF_FLOATS
        
        transf_offset = 0
        for short_body_name in outputs_to_track:
            body_part_name = next(k for k in pose_evals[i].keys() if short_body_name in k)
            for component_name, component_val in pose_evals[i][body_part_name].items():
                for dim_name, dim_val in component_val.items():
                    attrib_idx = attrib_ordering[component_name][dim_name]
                    y[subanim_idx][subanim_pos][attrib_idx] = dim_val
            transf_offset += TRANSF_FLOATS
    subanim_offset = subanim_idx

indices = np.arange(num_subanims)
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# ok time to train

RNN = layers.LSTM
HIDDEN_SIZE = 1024
BATCH_SIZE = 128
LAYERS = 1

print('build model...')

model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(SUBANIM_LEN, INPUT_ATTRIBS)))
model.add(layers.RepeatVector(SUBANIM_LEN))

for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(OUTPUT_ATTRIBS, activation='relu')))

model.compile(loss='mse',
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

print('starting training...')

for iteration in range(1, 100000):
    print('iteration: %d' % iteration)
    if(split_at == len(x)):
        model.fit(x_train, y_train, 
              batch_size=BATCH_SIZE,
              epochs=1)
    else:
        model.fit(x_train, y_train, 
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))

    if iteration % 5 == 0:
        model.save_weights("model.h5")