# vrcap
LSTM that predicts full body pose given the inputs from a vr setup: head, 
left/right hands, and root. To be used for an upcoming project, a low 
budget, full body mocap setup

Uses [pyfbx\_i42](https://github.com/ideasman42/pyfbx_i42) for fbx parsing

Issues:
* Forgot to account for fact that head/hand transforms for inputs are local to 
parent bone transform when training, need to convert inputs to global space
