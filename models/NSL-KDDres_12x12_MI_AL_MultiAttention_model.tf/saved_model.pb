ӧ	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
}
conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv0/kernel
v
 conv0/kernel/Read/ReadVariableOpReadVariableOpconv0/kernel*'
_output_shapes
:?*
dtype0
m

conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
conv0/bias
f
conv0/bias/Read/ReadVariableOpReadVariableOp
conv0/bias*
_output_shapes	
:?*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:Q@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
|
mc_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namemc_output/kernel
u
$mc_output/kernel/Read/ReadVariableOpReadVariableOpmc_output/kernel*
_output_shapes

:@*
dtype0
t
mc_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemc_output/bias
m
"mc_output/bias/Read/ReadVariableOpReadVariableOpmc_output/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv2d/kernel
y
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*(
_output_shapes
:??*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/conv0/kernel/m
?
'Adam/conv0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv0/kernel/m*'
_output_shapes
:?*
dtype0
{
Adam/conv0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/conv0/bias/m
t
%Adam/conv0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv0/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:Q@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/mc_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/mc_output/kernel/m
?
+Adam/mc_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mc_output/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/mc_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/mc_output/bias/m
{
)Adam/mc_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/mc_output/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*(
_output_shapes
:??*
dtype0
}
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/conv2d/bias/m
v
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_nameAdam/conv0/kernel/v
?
'Adam/conv0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv0/kernel/v*'
_output_shapes
:?*
dtype0
{
Adam/conv0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameAdam/conv0/bias/v
t
%Adam/conv0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv0/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:Q@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/mc_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_nameAdam/mc_output/kernel/v
?
+Adam/mc_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mc_output/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/mc_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/mc_output/bias/v
{
)Adam/mc_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/mc_output/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*(
_output_shapes
:??*
dtype0
}
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameAdam/conv2d/bias/v
v
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
]
	conv1
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
h

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
h

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemomp&mq'mr,ms-mt7mu8mvvwvx&vy'vz,v{-v|7v}8v~
8
0
1
72
83
&4
'5
,6
-7
 
8
0
1
72
83
&4
'5
,6
-7
?

trainable_variables
9layer_regularization_losses
:layer_metrics
;metrics
<non_trainable_variables
regularization_losses
	variables

=layers
 
XV
VARIABLE_VALUEconv0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
trainable_variables
>layer_regularization_losses
?layer_metrics
@metrics
Anon_trainable_variables
regularization_losses
	variables

Blayers
 
 
 
?
trainable_variables
Clayer_regularization_losses
Dlayer_metrics
Emetrics
Fnon_trainable_variables
regularization_losses
	variables

Glayers
h

7kernel
8bias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api

70
81
 

70
81
?
trainable_variables
Llayer_regularization_losses
Mlayer_metrics
Nmetrics
Onon_trainable_variables
regularization_losses
	variables

Players
 
 
 
?
trainable_variables
Qlayer_regularization_losses
Rlayer_metrics
Smetrics
Tnon_trainable_variables
regularization_losses
 	variables

Ulayers
 
 
 
?
"trainable_variables
Vlayer_regularization_losses
Wlayer_metrics
Xmetrics
Ynon_trainable_variables
#regularization_losses
$	variables

Zlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
?
(trainable_variables
[layer_regularization_losses
\layer_metrics
]metrics
^non_trainable_variables
)regularization_losses
*	variables

_layers
\Z
VARIABLE_VALUEmc_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEmc_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
?
.trainable_variables
`layer_regularization_losses
alayer_metrics
bmetrics
cnon_trainable_variables
/regularization_losses
0	variables

dlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 

e0
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 

70
81
 

70
81
?
Htrainable_variables
flayer_regularization_losses
glayer_metrics
hmetrics
inon_trainable_variables
Iregularization_losses
J	variables

jlayers
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ktotal
	lcount
m	variables
n	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

m	variables
{y
VARIABLE_VALUEAdam/conv0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/mc_output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/mc_output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/mc_output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/mc_output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv0/kernel
conv0/biasconv2d/kernelconv2d/biasdense/kernel
dense/biasmc_output/kernelmc_output/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_205817
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv0/kernel/Read/ReadVariableOpconv0/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp$mc_output/kernel/Read/ReadVariableOp"mc_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/conv0/kernel/m/Read/ReadVariableOp%Adam/conv0/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp+Adam/mc_output/kernel/m/Read/ReadVariableOp)Adam/mc_output/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp'Adam/conv0/kernel/v/Read/ReadVariableOp%Adam/conv0/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp+Adam/mc_output/kernel/v/Read/ReadVariableOp)Adam/mc_output/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_206221
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv0/kernel
conv0/biasdense/kernel
dense/biasmc_output/kernelmc_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biastotalcountAdam/conv0/kernel/mAdam/conv0/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/mc_output/kernel/mAdam/mc_output/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv0/kernel/vAdam/conv0/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/mc_output/kernel/vAdam/mc_output/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_206324??
?
?
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_206003
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/BiasAdd_1/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAddq
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2	
Sigmoidh
MulMulxconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Mul?
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D_1?
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp?
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAdd_1?
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????		?::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:S O
0
_output_shapes
:?????????		?

_user_specified_namex
?
D
(__inference_flatten_layer_call_fn_206065

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2055992
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????Q2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_205817
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_2054712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_205548
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/BiasAdd_1/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAddq
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2	
Sigmoidh
MulMulxconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Mul?
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D_1?
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp?
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAdd_1?
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????		?::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:S O
0
_output_shapes
:?????????		?

_user_specified_namex
?/
?
!__inference__wrapped_model_205471
input_1.
*model_conv0_conv2d_readvariableop_resource/
+model_conv0_biasadd_readvariableop_resource"
model_pixel_attention2d_205445"
model_pixel_attention2d_205447.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource2
.model_mc_output_matmul_readvariableop_resource3
/model_mc_output_biasadd_readvariableop_resource
identity??"model/conv0/BiasAdd/ReadVariableOp?!model/conv0/Conv2D/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?&model/mc_output/BiasAdd/ReadVariableOp?%model/mc_output/MatMul/ReadVariableOp?/model/pixel_attention2d/StatefulPartitionedCall?
!model/conv0/Conv2D/ReadVariableOpReadVariableOp*model_conv0_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02#
!model/conv0/Conv2D/ReadVariableOp?
model/conv0/Conv2DConv2Dinput_1)model/conv0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
model/conv0/Conv2D?
"model/conv0/BiasAdd/ReadVariableOpReadVariableOp+model_conv0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"model/conv0/BiasAdd/ReadVariableOp?
model/conv0/BiasAddBiasAddmodel/conv0/Conv2D:output:0*model/conv0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
model/conv0/BiasAdd?
model/conv0/ReluRelumodel/conv0/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
model/conv0/Relu?
model/dropout/IdentityIdentitymodel/conv0/Relu:activations:0*
T0*0
_output_shapes
:?????????		?2
model/dropout/Identity?
/model/pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCallmodel/dropout/Identity:output:0model_pixel_attention2d_205445model_pixel_attention2d_205447*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_call_20544421
/model/pixel_attention2d/StatefulPartitionedCall?
#model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#model/lambda/Mean/reduction_indices?
model/lambda/MeanMean8model/pixel_attention2d/StatefulPartitionedCall:output:0,model/lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
model/lambda/Mean?
model/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
model/lambda/ExpandDims/dim?
model/lambda/ExpandDims
ExpandDimsmodel/lambda/Mean:output:0$model/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2
model/lambda/ExpandDims{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????Q   2
model/flatten/Const?
model/flatten/ReshapeReshape model/lambda/ExpandDims:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????Q2
model/flatten/Reshape?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:Q@*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense/Relu?
%model/mc_output/MatMul/ReadVariableOpReadVariableOp.model_mc_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%model/mc_output/MatMul/ReadVariableOp?
model/mc_output/MatMulMatMulmodel/dense/Relu:activations:0-model/mc_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/mc_output/MatMul?
&model/mc_output/BiasAdd/ReadVariableOpReadVariableOp/model_mc_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/mc_output/BiasAdd/ReadVariableOp?
model/mc_output/BiasAddBiasAdd model/mc_output/MatMul:product:0.model/mc_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/mc_output/BiasAdd?
model/mc_output/SoftmaxSoftmax model/mc_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/mc_output/Softmax?
IdentityIdentity!model/mc_output/Softmax:softmax:0#^model/conv0/BiasAdd/ReadVariableOp"^model/conv0/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp'^model/mc_output/BiasAdd/ReadVariableOp&^model/mc_output/MatMul/ReadVariableOp0^model/pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"model/conv0/BiasAdd/ReadVariableOp"model/conv0/BiasAdd/ReadVariableOp2F
!model/conv0/Conv2D/ReadVariableOp!model/conv0/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2P
&model/mc_output/BiasAdd/ReadVariableOp&model/mc_output/BiasAdd/ReadVariableOp2N
%model/mc_output/MatMul/ReadVariableOp%model/mc_output/MatMul/ReadVariableOp2b
/model/pixel_attention2d/StatefulPartitionedCall/model/pixel_attention2d/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_206060

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????Q   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????Q2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????Q2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
{
&__inference_conv0_layer_call_fn_205960

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv0_layer_call_and_return_conditional_losses_2054862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
A__inference_model_layer_call_and_return_conditional_losses_205861

inputs(
$conv0_conv2d_readvariableop_resource)
%conv0_biasadd_readvariableop_resource
pixel_attention2d_205835
pixel_attention2d_205837(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource,
(mc_output_matmul_readvariableop_resource-
)mc_output_biasadd_readvariableop_resource
identity??conv0/BiasAdd/ReadVariableOp?conv0/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp? mc_output/BiasAdd/ReadVariableOp?mc_output/MatMul/ReadVariableOp?)pixel_attention2d/StatefulPartitionedCall?
conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv0/Conv2D/ReadVariableOp?
conv0/Conv2DConv2Dinputs#conv0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv0/Conv2D?
conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv0/BiasAdd/ReadVariableOp?
conv0/BiasAddBiasAddconv0/Conv2D:output:0$conv0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv0/BiasAdds

conv0/ReluReluconv0/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2

conv0/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/dropout/Const?
dropout/dropout/MulMulconv0/Relu:activations:0dropout/dropout/Const:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeconv0/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????		?*
dtype0*

seed2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????		?2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????		?2
dropout/dropout/Mul_1?
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCalldropout/dropout/Mul_1:z:0pixel_attention2d_205835pixel_attention2d_205837*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_call_2054442+
)pixel_attention2d/StatefulPartitionedCall?
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lambda/Mean/reduction_indices?
lambda/MeanMean2pixel_attention2d/StatefulPartitionedCall:output:0&lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
lambda/Meany
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lambda/ExpandDims/dim?
lambda/ExpandDims
ExpandDimslambda/Mean:output:0lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2
lambda/ExpandDimso
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????Q   2
flatten/Const?
flatten/ReshapeReshapelambda/ExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????Q2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Q@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
mc_output/MatMul/ReadVariableOpReadVariableOp(mc_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
mc_output/MatMul/ReadVariableOp?
mc_output/MatMulMatMuldense/Relu:activations:0'mc_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mc_output/MatMul?
 mc_output/BiasAdd/ReadVariableOpReadVariableOp)mc_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 mc_output/BiasAdd/ReadVariableOp?
mc_output/BiasAddBiasAddmc_output/MatMul:product:0(mc_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mc_output/BiasAdd
mc_output/SoftmaxSoftmaxmc_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
mc_output/Softmax?
IdentityIdentitymc_output/Softmax:softmax:0^conv0/BiasAdd/ReadVariableOp^conv0/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp!^mc_output/BiasAdd/ReadVariableOp ^mc_output/MatMul/ReadVariableOp*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2<
conv0/BiasAdd/ReadVariableOpconv0/BiasAdd/ReadVariableOp2:
conv0/Conv2D/ReadVariableOpconv0/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2D
 mc_output/BiasAdd/ReadVariableOp mc_output/BiasAdd/ReadVariableOp2B
mc_output/MatMul/ReadVariableOpmc_output/MatMul/ReadVariableOp2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_conv0_layer_call_and_return_conditional_losses_205951

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

*__inference_mc_output_layer_call_fn_206105

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_mc_output_layer_call_and_return_conditional_losses_2056452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_205599

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????Q   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????Q2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????Q2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????		:W S
/
_output_shapes
:?????????		
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_205514

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????		?*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????		?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????		?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_205786
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2057672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
A__inference_model_layer_call_and_return_conditional_losses_205689
input_1
conv0_205665
conv0_205667
pixel_attention2d_205671
pixel_attention2d_205673
dense_205678
dense_205680
mc_output_205683
mc_output_205685
identity??conv0/StatefulPartitionedCall?dense/StatefulPartitionedCall?!mc_output/StatefulPartitionedCall?)pixel_attention2d/StatefulPartitionedCall?
conv0/StatefulPartitionedCallStatefulPartitionedCallinput_1conv0_205665conv0_205667*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv0_layer_call_and_return_conditional_losses_2054862
conv0/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2055192
dropout/PartitionedCall?
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pixel_attention2d_205671pixel_attention2d_205673*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_2055482+
)pixel_attention2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2055802
lambda/PartitionedCall?
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2055992
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_205678dense_205680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2056182
dense/StatefulPartitionedCall?
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_205683mc_output_205685*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_mc_output_layer_call_and_return_conditional_losses_2056452#
!mc_output/StatefulPartitionedCall?
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_206044

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?	
?
E__inference_mc_output_layer_call_and_return_conditional_losses_206096

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_206076

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Q@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Q::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Q
 
_user_specified_nameinputs
?*
?
A__inference_model_layer_call_and_return_conditional_losses_205898

inputs(
$conv0_conv2d_readvariableop_resource)
%conv0_biasadd_readvariableop_resource
pixel_attention2d_205872
pixel_attention2d_205874(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource,
(mc_output_matmul_readvariableop_resource-
)mc_output_biasadd_readvariableop_resource
identity??conv0/BiasAdd/ReadVariableOp?conv0/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp? mc_output/BiasAdd/ReadVariableOp?mc_output/MatMul/ReadVariableOp?)pixel_attention2d/StatefulPartitionedCall?
conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv0/Conv2D/ReadVariableOp?
conv0/Conv2DConv2Dinputs#conv0/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv0/Conv2D?
conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv0/BiasAdd/ReadVariableOp?
conv0/BiasAddBiasAddconv0/Conv2D:output:0$conv0/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv0/BiasAdds

conv0/ReluReluconv0/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2

conv0/Relu?
dropout/IdentityIdentityconv0/Relu:activations:0*
T0*0
_output_shapes
:?????????		?2
dropout/Identity?
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCalldropout/Identity:output:0pixel_attention2d_205872pixel_attention2d_205874*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? * 
fR
__inference_call_2054442+
)pixel_attention2d/StatefulPartitionedCall?
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lambda/Mean/reduction_indices?
lambda/MeanMean2pixel_attention2d/StatefulPartitionedCall:output:0&lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
lambda/Meany
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
lambda/ExpandDims/dim?
lambda/ExpandDims
ExpandDimslambda/Mean:output:0lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2
lambda/ExpandDimso
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????Q   2
flatten/Const?
flatten/ReshapeReshapelambda/ExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????Q2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:Q@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
mc_output/MatMul/ReadVariableOpReadVariableOp(mc_output_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02!
mc_output/MatMul/ReadVariableOp?
mc_output/MatMulMatMuldense/Relu:activations:0'mc_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mc_output/MatMul?
 mc_output/BiasAdd/ReadVariableOpReadVariableOp)mc_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 mc_output/BiasAdd/ReadVariableOp?
mc_output/BiasAddBiasAddmc_output/MatMul:product:0(mc_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
mc_output/BiasAdd
mc_output/SoftmaxSoftmaxmc_output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
mc_output/Softmax?
IdentityIdentitymc_output/Softmax:softmax:0^conv0/BiasAdd/ReadVariableOp^conv0/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp!^mc_output/BiasAdd/ReadVariableOp ^mc_output/MatMul/ReadVariableOp*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2<
conv0/BiasAdd/ReadVariableOpconv0/BiasAdd/ReadVariableOp2:
conv0/Conv2D/ReadVariableOpconv0/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2D
 mc_output/BiasAdd/ReadVariableOp mc_output/BiasAdd/ReadVariableOp2B
mc_output/MatMul/ReadVariableOpmc_output/MatMul/ReadVariableOp2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_205580

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
A__inference_model_layer_call_and_return_conditional_losses_205767

inputs
conv0_205743
conv0_205745
pixel_attention2d_205749
pixel_attention2d_205751
dense_205756
dense_205758
mc_output_205761
mc_output_205763
identity??conv0/StatefulPartitionedCall?dense/StatefulPartitionedCall?!mc_output/StatefulPartitionedCall?)pixel_attention2d/StatefulPartitionedCall?
conv0/StatefulPartitionedCallStatefulPartitionedCallinputsconv0_205743conv0_205745*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv0_layer_call_and_return_conditional_losses_2054862
conv0/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2055192
dropout/PartitionedCall?
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pixel_attention2d_205749pixel_attention2d_205751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_2055482+
)pixel_attention2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2055802
lambda/PartitionedCall?
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2055992
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_205756dense_205758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2056182
dense/StatefulPartitionedCall?
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_205761mc_output_205763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_mc_output_layer_call_and_return_conditional_losses_2056452#
!mc_output/StatefulPartitionedCall?
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_205940

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2057672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
2__inference_pixel_attention2d_layer_call_fn_206012
x
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_2055482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????		?::22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????		?

_user_specified_namex
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_205519

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????		?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????		?2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
__inference_call_205444
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/BiasAdd_1/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAddq
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2	
Sigmoidh
MulMulxconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Mul?
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D_1?
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp?
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAdd_1?
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????		?::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:S O
0
_output_shapes
:?????????		?

_user_specified_namex
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_206036

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?

?
A__inference_conv0_layer_call_and_return_conditional_losses_205486

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_205572

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:?????????		2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim?

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????		2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
D
(__inference_dropout_layer_call_fn_205987

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2055192
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?	
?
A__inference_dense_layer_call_and_return_conditional_losses_205618

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Q@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Q::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Q
 
_user_specified_nameinputs
?
C
'__inference_lambda_layer_call_fn_206054

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2055802
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?!
?
A__inference_model_layer_call_and_return_conditional_losses_205719

inputs
conv0_205695
conv0_205697
pixel_attention2d_205701
pixel_attention2d_205703
dense_205708
dense_205710
mc_output_205713
mc_output_205715
identity??conv0/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!mc_output/StatefulPartitionedCall?)pixel_attention2d/StatefulPartitionedCall?
conv0/StatefulPartitionedCallStatefulPartitionedCallinputsconv0_205695conv0_205697*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv0_layer_call_and_return_conditional_losses_2054862
conv0/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2055142!
dropout/StatefulPartitionedCall?
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pixel_attention2d_205701pixel_attention2d_205703*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_2055482+
)pixel_attention2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2055722
lambda/PartitionedCall?
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2055992
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_205708dense_205710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2056182
dense/StatefulPartitionedCall?
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_205713mc_output_205715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_mc_output_layer_call_and_return_conditional_losses_2056452#
!mc_output/StatefulPartitionedCall?
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
A__inference_model_layer_call_and_return_conditional_losses_205662
input_1
conv0_205497
conv0_205499
pixel_attention2d_205559
pixel_attention2d_205561
dense_205629
dense_205631
mc_output_205656
mc_output_205658
identity??conv0/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!mc_output/StatefulPartitionedCall?)pixel_attention2d/StatefulPartitionedCall?
conv0/StatefulPartitionedCallStatefulPartitionedCallinput_1conv0_205497conv0_205499*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv0_layer_call_and_return_conditional_losses_2054862
conv0/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2055142!
dropout/StatefulPartitionedCall?
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pixel_attention2d_205559pixel_attention2d_205561*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_2055482+
)pixel_attention2d/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2055722
lambda/PartitionedCall?
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2055992
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_205629dense_205631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2056182
dense/StatefulPartitionedCall?
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_205656mc_output_205658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_mc_output_layer_call_and_return_conditional_losses_2056452#
!mc_output/StatefulPartitionedCall?
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
(__inference_dropout_layer_call_fn_205982

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????		?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2055142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
C
'__inference_lambda_layer_call_fn_206049

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2055722
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????		2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_205972

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????		?*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????		?2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????		?2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????		?2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_205919

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2057192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_205977

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????		?2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????		?2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????		?:X T
0
_output_shapes
:?????????		?
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_205738
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2057192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?	
?
E__inference_mc_output_layer_call_and_return_conditional_losses_205645

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?E
?
__inference__traced_save_206221
file_prefix+
'savev2_conv0_kernel_read_readvariableop)
%savev2_conv0_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop/
+savev2_mc_output_kernel_read_readvariableop-
)savev2_mc_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_conv0_kernel_m_read_readvariableop0
,savev2_adam_conv0_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop6
2savev2_adam_mc_output_kernel_m_read_readvariableop4
0savev2_adam_mc_output_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop2
.savev2_adam_conv0_kernel_v_read_readvariableop0
,savev2_adam_conv0_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop6
2savev2_adam_mc_output_kernel_v_read_readvariableop4
0savev2_adam_mc_output_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv0_kernel_read_readvariableop%savev2_conv0_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop+savev2_mc_output_kernel_read_readvariableop)savev2_mc_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_conv0_kernel_m_read_readvariableop,savev2_adam_conv0_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop2savev2_adam_mc_output_kernel_m_read_readvariableop0savev2_adam_mc_output_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop.savev2_adam_conv0_kernel_v_read_readvariableop,savev2_adam_conv0_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop2savev2_adam_mc_output_kernel_v_read_readvariableop0savev2_adam_mc_output_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:Q@:@:@:: : : : : :??:?: : :?:?:Q@:@:@::??:?:?:?:Q@:@:@::??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:!

_output_shapes	
:?:$ 

_output_shapes

:Q@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?:!

_output_shapes	
:?:$ 

_output_shapes

:Q@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?:!

_output_shapes	
:?:$ 

_output_shapes

:Q@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::.*
(
_output_shapes
:??:!

_output_shapes	
:?: 

_output_shapes
: 
?
?
__inference_call_206028
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/BiasAdd_1/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAddq
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2	
Sigmoidh
MulMulxconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:?????????		?2
Mul?
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d/Conv2D_1/ReadVariableOp?
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?*
paddingVALID*
strides
2
conv2d/Conv2D_1?
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp?
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????		?2
conv2d/BiasAdd_1?
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*0
_output_shapes
:?????????		?2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????		?::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:S O
0
_output_shapes
:?????????		?

_user_specified_namex
??
?
"__inference__traced_restore_206324
file_prefix!
assignvariableop_conv0_kernel!
assignvariableop_1_conv0_bias#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias'
#assignvariableop_4_mc_output_kernel%
!assignvariableop_5_mc_output_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate%
!assignvariableop_11_conv2d_kernel#
assignvariableop_12_conv2d_bias
assignvariableop_13_total
assignvariableop_14_count+
'assignvariableop_15_adam_conv0_kernel_m)
%assignvariableop_16_adam_conv0_bias_m+
'assignvariableop_17_adam_dense_kernel_m)
%assignvariableop_18_adam_dense_bias_m/
+assignvariableop_19_adam_mc_output_kernel_m-
)assignvariableop_20_adam_mc_output_bias_m,
(assignvariableop_21_adam_conv2d_kernel_m*
&assignvariableop_22_adam_conv2d_bias_m+
'assignvariableop_23_adam_conv0_kernel_v)
%assignvariableop_24_adam_conv0_bias_v+
'assignvariableop_25_adam_dense_kernel_v)
%assignvariableop_26_adam_dense_bias_v/
+assignvariableop_27_adam_mc_output_kernel_v-
)assignvariableop_28_adam_mc_output_bias_v,
(assignvariableop_29_adam_conv2d_kernel_v*
&assignvariableop_30_adam_conv2d_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_mc_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_mc_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_conv2d_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_conv0_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_conv0_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_mc_output_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_mc_output_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_conv2d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_conv2d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_conv0_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_conv0_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_mc_output_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_mc_output_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv2d_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv2d_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*?
_input_shapes?
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
{
&__inference_dense_layer_call_fn_206085

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2056182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????Q::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Q
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????=
	mc_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?>
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?;
_tf_keras_network?;{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv0", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0701895119470261, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv0", 0, 0, {}]]]}, {"class_name": "PixelAttention2D", "config": {"name": "pixel_attention2d", "trainable": true, "dtype": "float32", "Att_filters": 512}, "name": "pixel_attention2d", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxIAAAB0AHQBfABkA2QCjQJkBIMCUwApBU7pAQAAACkB2gRh\neGlz6f////9yAwAAACkC2gtleHBhbmRfZGltc9oLcmVkdWNlX21lYW4pAdoBeKkAcgcAAAD6My9o\nb21lL29tZW4tZ2l1c3kvQXR0ZW50aW9uL1hBSS9UcmFpblNpbmdsZU91dHB1dC5wedoIPGxhbWJk\nYT7eAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "XAI.TrainSingleOutput", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["pixel_attention2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mc_output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mc_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["mc_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv0", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0701895119470261, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv0", 0, 0, {}]]]}, {"class_name": "PixelAttention2D", "config": {"name": "pixel_attention2d", "trainable": true, "dtype": "float32", "Att_filters": 512}, "name": "pixel_attention2d", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxIAAAB0AHQBfABkA2QCjQJkBIMCUwApBU7pAQAAACkB2gRh\neGlz6f////9yAwAAACkC2gtleHBhbmRfZGltc9oLcmVkdWNlX21lYW4pAdoBeKkAcgcAAAD6My9o\nb21lL29tZW4tZ2l1c3kvQXR0ZW50aW9uL1hBSS9UcmFpblNpbmdsZU91dHB1dC5wedoIPGxhbWJk\nYT7eAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "XAI.TrainSingleOutput", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["pixel_attention2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mc_output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mc_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["mc_output", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": [0.5111425885041833, 0.48885741149581674], "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0007519648061133921, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 1]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0701895119470261, "noise_shape": null, "seed": null}}
?
	conv1
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?call"?
_tf_keras_layer?{"class_name": "PixelAttention2D", "name": "pixel_attention2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_attention2d", "trainable": true, "dtype": "float32", "Att_filters": 512}}
?
trainable_variables
regularization_losses
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxIAAAB0AHQBfABkA2QCjQJkBIMCUwApBU7pAQAAACkB2gRh\neGlz6f////9yAwAAACkC2gtleHBhbmRfZGltc9oLcmVkdWNlX21lYW4pAdoBeKkAcgcAAAD6My9o\nb21lL29tZW4tZ2l1c3kvQXR0ZW50aW9uL1hBSS9UcmFpblNpbmdsZU91dHB1dC5wedoIPGxhbWJk\nYT7eAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "XAI.TrainSingleOutput", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?
"trainable_variables
#regularization_losses
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 81}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81]}}
?

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "mc_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "mc_output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemomp&mq'mr,ms-mt7mu8mvvwvx&vy'vz,v{-v|7v}8v~"
	optimizer
X
0
1
72
83
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
72
83
&4
'5
,6
-7"
trackable_list_wrapper
?

trainable_variables
9layer_regularization_losses
:layer_metrics
;metrics
<non_trainable_variables
regularization_losses
	variables

=layers
__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
':%?2conv0/kernel
:?2
conv0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
trainable_variables
>layer_regularization_losses
?layer_metrics
@metrics
Anon_trainable_variables
regularization_losses
	variables

Blayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Clayer_regularization_losses
Dlayer_metrics
Emetrics
Fnon_trainable_variables
regularization_losses
	variables

Glayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

7kernel
8bias
Htrainable_variables
Iregularization_losses
J	variables
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9, 9, 512]}}
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
trainable_variables
Llayer_regularization_losses
Mlayer_metrics
Nmetrics
Onon_trainable_variables
regularization_losses
	variables

Players
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Qlayer_regularization_losses
Rlayer_metrics
Smetrics
Tnon_trainable_variables
regularization_losses
 	variables

Ulayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"trainable_variables
Vlayer_regularization_losses
Wlayer_metrics
Xmetrics
Ynon_trainable_variables
#regularization_losses
$	variables

Zlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:Q@2dense/kernel
:@2
dense/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
(trainable_variables
[layer_regularization_losses
\layer_metrics
]metrics
^non_trainable_variables
)regularization_losses
*	variables

_layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": @2mc_output/kernel
:2mc_output/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
.trainable_variables
`layer_regularization_losses
alayer_metrics
bmetrics
cnon_trainable_variables
/regularization_losses
0	variables

dlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'??2conv2d/kernel
:?2conv2d/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
e0"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
Htrainable_variables
flayer_regularization_losses
glayer_metrics
hmetrics
inon_trainable_variables
Iregularization_losses
J	variables

jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	ktotal
	lcount
m	variables
n	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
,:*?2Adam/conv0/kernel/m
:?2Adam/conv0/bias/m
#:!Q@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
':%@2Adam/mc_output/kernel/m
!:2Adam/mc_output/bias/m
.:,??2Adam/conv2d/kernel/m
:?2Adam/conv2d/bias/m
,:*?2Adam/conv0/kernel/v
:?2Adam/conv0/bias/v
#:!Q@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
':%@2Adam/mc_output/kernel/v
!:2Adam/mc_output/bias/v
.:,??2Adam/conv2d/kernel/v
:?2Adam/conv2d/bias/v
?2?
&__inference_model_layer_call_fn_205786
&__inference_model_layer_call_fn_205940
&__inference_model_layer_call_fn_205738
&__inference_model_layer_call_fn_205919?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_205471?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
A__inference_model_layer_call_and_return_conditional_losses_205861
A__inference_model_layer_call_and_return_conditional_losses_205898
A__inference_model_layer_call_and_return_conditional_losses_205689
A__inference_model_layer_call_and_return_conditional_losses_205662?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_conv0_layer_call_fn_205960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv0_layer_call_and_return_conditional_losses_205951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_205982
(__inference_dropout_layer_call_fn_205987?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_205972
C__inference_dropout_layer_call_and_return_conditional_losses_205977?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
2__inference_pixel_attention2d_layer_call_fn_206012?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_206003?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_call_206028?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_lambda_layer_call_fn_206049
'__inference_lambda_layer_call_fn_206054?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_lambda_layer_call_and_return_conditional_losses_206044
B__inference_lambda_layer_call_and_return_conditional_losses_206036?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_flatten_layer_call_fn_206065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_206060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_206085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_206076?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_mc_output_layer_call_fn_206105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_mc_output_layer_call_and_return_conditional_losses_206096?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_205817input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_205471{78&',-8?5
.?+
)?&
input_1?????????
? "5?2
0
	mc_output#? 
	mc_output?????????w
__inference_call_206028\783?0
)?&
$?!
x?????????		?
? "!??????????		??
A__inference_conv0_layer_call_and_return_conditional_losses_205951m7?4
-?*
(?%
inputs?????????
? ".?+
$?!
0?????????		?
? ?
&__inference_conv0_layer_call_fn_205960`7?4
-?*
(?%
inputs?????????
? "!??????????		??
A__inference_dense_layer_call_and_return_conditional_losses_206076\&'/?,
%?"
 ?
inputs?????????Q
? "%?"
?
0?????????@
? y
&__inference_dense_layer_call_fn_206085O&'/?,
%?"
 ?
inputs?????????Q
? "??????????@?
C__inference_dropout_layer_call_and_return_conditional_losses_205972n<?9
2?/
)?&
inputs?????????		?
p
? ".?+
$?!
0?????????		?
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_205977n<?9
2?/
)?&
inputs?????????		?
p 
? ".?+
$?!
0?????????		?
? ?
(__inference_dropout_layer_call_fn_205982a<?9
2?/
)?&
inputs?????????		?
p
? "!??????????		??
(__inference_dropout_layer_call_fn_205987a<?9
2?/
)?&
inputs?????????		?
p 
? "!??????????		??
C__inference_flatten_layer_call_and_return_conditional_losses_206060`7?4
-?*
(?%
inputs?????????		
? "%?"
?
0?????????Q
? 
(__inference_flatten_layer_call_fn_206065S7?4
-?*
(?%
inputs?????????		
? "??????????Q?
B__inference_lambda_layer_call_and_return_conditional_losses_206036q@?=
6?3
)?&
inputs?????????		?

 
p
? "-?*
#? 
0?????????		
? ?
B__inference_lambda_layer_call_and_return_conditional_losses_206044q@?=
6?3
)?&
inputs?????????		?

 
p 
? "-?*
#? 
0?????????		
? ?
'__inference_lambda_layer_call_fn_206049d@?=
6?3
)?&
inputs?????????		?

 
p
? " ??????????		?
'__inference_lambda_layer_call_fn_206054d@?=
6?3
)?&
inputs?????????		?

 
p 
? " ??????????		?
E__inference_mc_output_layer_call_and_return_conditional_losses_206096\,-/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? }
*__inference_mc_output_layer_call_fn_206105O,-/?,
%?"
 ?
inputs?????????@
? "???????????
A__inference_model_layer_call_and_return_conditional_losses_205662s78&',-@?=
6?3
)?&
input_1?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_205689s78&',-@?=
6?3
)?&
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_205861r78&',-??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_205898r78&',-??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_layer_call_fn_205738f78&',-@?=
6?3
)?&
input_1?????????
p

 
? "???????????
&__inference_model_layer_call_fn_205786f78&',-@?=
6?3
)?&
input_1?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_205919e78&',-??<
5?2
(?%
inputs?????????
p

 
? "???????????
&__inference_model_layer_call_fn_205940e78&',-??<
5?2
(?%
inputs?????????
p 

 
? "???????????
M__inference_pixel_attention2d_layer_call_and_return_conditional_losses_206003i783?0
)?&
$?!
x?????????		?
? ".?+
$?!
0?????????		?
? ?
2__inference_pixel_attention2d_layer_call_fn_206012\783?0
)?&
$?!
x?????????		?
? "!??????????		??
$__inference_signature_wrapper_205817?78&',-C?@
? 
9?6
4
input_1)?&
input_1?????????"5?2
0
	mc_output#? 
	mc_output?????????