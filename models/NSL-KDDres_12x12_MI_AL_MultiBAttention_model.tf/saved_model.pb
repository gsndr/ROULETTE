õ
¾
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8Ô

|
conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv0/kernel
u
 conv0/kernel/Read/ReadVariableOpReadVariableOpconv0/kernel*&
_output_shapes
:@*
dtype0
l

conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv0/bias
e
conv0/bias/Read/ReadVariableOpReadVariableOp
conv0/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	y*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	y*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À<*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
À<*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
}
mc_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namemc_output/kernel
v
$mc_output/kernel/Read/ReadVariableOpReadVariableOpmc_output/kernel*
_output_shapes
:	*
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
{
b_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameb_output/kernel
t
#b_output/kernel/Read/ReadVariableOpReadVariableOpb_output/kernel*
_output_shapes
:	*
dtype0
r
b_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameb_output/bias
k
!b_output/bias/Read/ReadVariableOpReadVariableOpb_output/bias*
_output_shapes
:*
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
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/conv0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/conv0/kernel/m

'Adam/conv0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv0/kernel/m*&
_output_shapes
:@*
dtype0
z
Adam/conv0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/conv0/bias/m
s
%Adam/conv0/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv0/bias/m*
_output_shapes
:@*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	y*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	y*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À<*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
À<*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/mc_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/mc_output/kernel/m

+Adam/mc_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mc_output/kernel/m*
_output_shapes
:	*
dtype0

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

Adam/b_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/b_output/kernel/m

*Adam/b_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/b_output/kernel/m*
_output_shapes
:	*
dtype0

Adam/b_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/b_output/bias/m
y
(Adam/b_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/b_output/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0

Adam/conv0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/conv0/kernel/v

'Adam/conv0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv0/kernel/v*&
_output_shapes
:@*
dtype0
z
Adam/conv0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/conv0/bias/v
s
%Adam/conv0/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv0/bias/v*
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	y*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	y*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
À<*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
À<*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/mc_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameAdam/mc_output/kernel/v

+Adam/mc_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mc_output/kernel/v*
_output_shapes
:	*
dtype0

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

Adam/b_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/b_output/kernel/v

*Adam/b_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/b_output/kernel/v*
_output_shapes
:	*
dtype0

Adam/b_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/b_output/bias/v
y
(Adam/b_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/b_output/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
­M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*èL
valueÞLBÛL BÔL
§
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
loss
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
]
	conv1
trainable_variables
	variables
 regularization_losses
!	keras_api
R
"trainable_variables
#	variables
$regularization_losses
%	keras_api
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
h

@kernel
Abias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
°
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratemm.m/m4m 5m¡:m¢;m£@m¤Am¥Km¦Lm§v¨v©.vª/v«4v¬5v­:v®;v¯@v°Av±Kv²Lv³
 
V
0
1
K2
L3
.4
/5
46
57
:8
;9
@10
A11
V
0
1
K2
L3
.4
/5
46
57
:8
;9
@10
A11
 
­
trainable_variables
Mlayer_regularization_losses
	variables
Nlayer_metrics
regularization_losses

Olayers
Pnon_trainable_variables
Qmetrics
 
XV
VARIABLE_VALUEconv0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
Rlayer_regularization_losses
trainable_variables
	variables
Slayer_metrics
regularization_losses

Tlayers
Unon_trainable_variables
Vmetrics
 
 
 
­
Wlayer_regularization_losses
trainable_variables
	variables
Xlayer_metrics
regularization_losses

Ylayers
Znon_trainable_variables
[metrics
h

Kkernel
Lbias
\trainable_variables
]	variables
^regularization_losses
_	keras_api

K0
L1

K0
L1
 
­
`layer_regularization_losses
trainable_variables
	variables
alayer_metrics
 regularization_losses

blayers
cnon_trainable_variables
dmetrics
 
 
 
­
elayer_regularization_losses
"trainable_variables
#	variables
flayer_metrics
$regularization_losses

glayers
hnon_trainable_variables
imetrics
 
 
 
­
jlayer_regularization_losses
&trainable_variables
'	variables
klayer_metrics
(regularization_losses

llayers
mnon_trainable_variables
nmetrics
 
 
 
­
olayer_regularization_losses
*trainable_variables
+	variables
player_metrics
,regularization_losses

qlayers
rnon_trainable_variables
smetrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
­
tlayer_regularization_losses
0trainable_variables
1	variables
ulayer_metrics
2regularization_losses

vlayers
wnon_trainable_variables
xmetrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
­
ylayer_regularization_losses
6trainable_variables
7	variables
zlayer_metrics
8regularization_losses

{layers
|non_trainable_variables
}metrics
\Z
VARIABLE_VALUEmc_output/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEmc_output/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
°
~layer_regularization_losses
<trainable_variables
=	variables
layer_metrics
>regularization_losses
layers
non_trainable_variables
metrics
[Y
VARIABLE_VALUEb_output/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEb_output/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

@0
A1
 
²
 layer_regularization_losses
Btrainable_variables
C	variables
layer_metrics
Dregularization_losses
layers
non_trainable_variables
metrics
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
N
0
1
2
3
4
5
6
7
	8

9
10
 

0
1
2
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
K0
L1

K0
L1
 
²
 layer_regularization_losses
\trainable_variables
]	variables
layer_metrics
^regularization_losses
layers
non_trainable_variables
metrics
 
 

0
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
8

total

count
	variables
	keras_api
8

total

count
	variables
	keras_api
8

total

count
	variables
	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
{y
VARIABLE_VALUEAdam/conv0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/mc_output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/mc_output/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_output/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/b_output/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/mc_output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/mc_output/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/b_output/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/b_output/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv0/kernel
conv0/biasconv2d/kernelconv2d/biasdense_1/kerneldense_1/biasdense/kernel
dense/biasb_output/kernelb_output/biasmc_output/kernelmc_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1443636
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¦
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv0/kernel/Read/ReadVariableOpconv0/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp$mc_output/kernel/Read/ReadVariableOp"mc_output/bias/Read/ReadVariableOp#b_output/kernel/Read/ReadVariableOp!b_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp'Adam/conv0/kernel/m/Read/ReadVariableOp%Adam/conv0/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp+Adam/mc_output/kernel/m/Read/ReadVariableOp)Adam/mc_output/bias/m/Read/ReadVariableOp*Adam/b_output/kernel/m/Read/ReadVariableOp(Adam/b_output/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp'Adam/conv0/kernel/v/Read/ReadVariableOp%Adam/conv0/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp+Adam/mc_output/kernel/v/Read/ReadVariableOp)Adam/mc_output/bias/v/Read/ReadVariableOp*Adam/b_output/kernel/v/Read/ReadVariableOp(Adam/b_output/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1444194
õ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv0/kernel
conv0/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasmc_output/kernelmc_output/biasb_output/kernelb_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biastotalcounttotal_1count_1total_2count_2Adam/conv0/kernel/mAdam/conv0/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/mc_output/kernel/mAdam/mc_output/bias/mAdam/b_output/kernel/mAdam/b_output/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv0/kernel/vAdam/conv0/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/mc_output/kernel/vAdam/mc_output/bias/vAdam/b_output/kernel/vAdam/b_output/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v*;
Tin4
220*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1444345Ä


__inference_call_1443111
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/BiasAdd_1/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d/Conv2D_1/ReadVariableOpª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAddp
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoidg
MulMulxconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Mul®
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d/Conv2D_1/ReadVariableOpÀ
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D_1¥
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp¬
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAdd_1÷
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
³1
ö
B__inference_model_layer_call_and_return_conditional_losses_1443496

inputs
conv0_1443460
conv0_1443462
pixel_attention2d_1443466
pixel_attention2d_1443468
dense_1_1443474
dense_1_1443476
dense_1443479
dense_1443481
b_output_1443484
b_output_1443486
mc_output_1443489
mc_output_1443491
identity

identity_1¢ b_output/StatefulPartitionedCall¢conv0/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!mc_output/StatefulPartitionedCall¢)pixel_attention2d/StatefulPartitionedCall
conv0/StatefulPartitionedCallStatefulPartitionedCallinputsconv0_1443460conv0_1443462*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_14431702
conv0/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_14431982!
dropout/StatefulPartitionedCallî
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pixel_attention2d_1443466pixel_attention2d_1443468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_14432322+
)pixel_attention2d/StatefulPartitionedCall
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_14432562
lambda/PartitionedCallû
flatten_1/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_14432832
flatten_1/PartitionedCallë
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_14432972
flatten/PartitionedCall¯
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1443474dense_1_1443476*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14433162!
dense_1/StatefulPartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1443479dense_1443481*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_14433432
dense/StatefulPartitionedCall¹
 b_output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0b_output_1443484b_output_1443486*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_b_output_layer_call_and_return_conditional_losses_14433702"
 b_output/StatefulPartitionedCall¼
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_1443489mc_output_1443491*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_mc_output_layer_call_and_return_conditional_losses_14433972#
!mc_output/StatefulPartitionedCallõ
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityø

Identity_1Identity)b_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2D
 b_output/StatefulPartitionedCall b_output/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è

¢
%__inference_signature_wrapper_1443636
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_14431552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ô	
Þ
E__inference_b_output_layer_call_and_return_conditional_losses_1444020

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_1_layer_call_and_return_conditional_losses_1443316

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À<*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<
 
_user_specified_nameinputs
Í

Û
B__inference_conv0_layer_call_and_return_conditional_losses_1443824

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
D
(__inference_lambda_layer_call_fn_1443922

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_14432562
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
'__inference_model_layer_call_fn_1443595
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_14435662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ä
_
C__inference_lambda_layer_call_and_return_conditional_losses_1443264

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ç
b
D__inference_dropout_layer_call_and_return_conditional_losses_1443203

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
Á
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_1443232
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/BiasAdd_1/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d/Conv2D_1/ReadVariableOpª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAddp
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoidg
MulMulxconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Mul®
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d/Conv2D_1/ReadVariableOpÀ
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D_1¥
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp¬
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAdd_1÷
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
º
`
D__inference_flatten_layer_call_and_return_conditional_losses_1443933

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿy   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
ß
F__inference_mc_output_layer_call_and_return_conditional_losses_1444000

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò	
Û
B__inference_dense_layer_call_and_return_conditional_losses_1443960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	y*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿy::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 
_user_specified_nameinputs
ùÃ
Ó
#__inference__traced_restore_1444345
file_prefix!
assignvariableop_conv0_kernel!
assignvariableop_1_conv0_bias#
assignvariableop_2_dense_kernel!
assignvariableop_3_dense_bias%
!assignvariableop_4_dense_1_kernel#
assignvariableop_5_dense_1_bias'
#assignvariableop_6_mc_output_kernel%
!assignvariableop_7_mc_output_bias&
"assignvariableop_8_b_output_kernel$
 assignvariableop_9_b_output_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate%
!assignvariableop_15_conv2d_kernel#
assignvariableop_16_conv2d_bias
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1
assignvariableop_21_total_2
assignvariableop_22_count_2+
'assignvariableop_23_adam_conv0_kernel_m)
%assignvariableop_24_adam_conv0_bias_m+
'assignvariableop_25_adam_dense_kernel_m)
%assignvariableop_26_adam_dense_bias_m-
)assignvariableop_27_adam_dense_1_kernel_m+
'assignvariableop_28_adam_dense_1_bias_m/
+assignvariableop_29_adam_mc_output_kernel_m-
)assignvariableop_30_adam_mc_output_bias_m.
*assignvariableop_31_adam_b_output_kernel_m,
(assignvariableop_32_adam_b_output_bias_m,
(assignvariableop_33_adam_conv2d_kernel_m*
&assignvariableop_34_adam_conv2d_bias_m+
'assignvariableop_35_adam_conv0_kernel_v)
%assignvariableop_36_adam_conv0_bias_v+
'assignvariableop_37_adam_dense_kernel_v)
%assignvariableop_38_adam_dense_bias_v-
)assignvariableop_39_adam_dense_1_kernel_v+
'assignvariableop_40_adam_dense_1_bias_v/
+assignvariableop_41_adam_mc_output_kernel_v-
)assignvariableop_42_adam_mc_output_bias_v.
*assignvariableop_43_adam_b_output_kernel_v,
(assignvariableop_44_adam_b_output_bias_v,
(assignvariableop_45_adam_conv2d_kernel_v*
&assignvariableop_46_adam_conv2d_bias_v
identity_48¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesî
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ö
_output_shapesÃ
À::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¤
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¦
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_mc_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_mc_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_b_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_b_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14®
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16§
AssignVariableOp_16AssignVariableOpassignvariableop_16_conv2d_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¡
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22£
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¯
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_conv0_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24­
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_conv0_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¯
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26­
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27±
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28¯
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_mc_output_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_mc_output_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31²
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_b_output_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32°
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_b_output_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33°
AssignVariableOp_33AssignVariableOp(assignvariableop_33_adam_conv2d_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34®
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_conv2d_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¯
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_conv0_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36­
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_conv0_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¯
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38­
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_1_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¯
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_1_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_mc_output_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_mc_output_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43²
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_b_output_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44°
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_b_output_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45°
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_conv2d_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46®
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_conv2d_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpè
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47Û
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*Ó
_input_shapesÁ
¾: :::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
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
û	
ß
F__inference_mc_output_layer_call_and_return_conditional_losses_1443397

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
_
C__inference_lambda_layer_call_and_return_conditional_losses_1443917

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
á/
Ô
B__inference_model_layer_call_and_return_conditional_losses_1443566

inputs
conv0_1443530
conv0_1443532
pixel_attention2d_1443536
pixel_attention2d_1443538
dense_1_1443544
dense_1_1443546
dense_1443549
dense_1443551
b_output_1443554
b_output_1443556
mc_output_1443559
mc_output_1443561
identity

identity_1¢ b_output/StatefulPartitionedCall¢conv0/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!mc_output/StatefulPartitionedCall¢)pixel_attention2d/StatefulPartitionedCall
conv0/StatefulPartitionedCallStatefulPartitionedCallinputsconv0_1443530conv0_1443532*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_14431702
conv0/StatefulPartitionedCallú
dropout/PartitionedCallPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_14432032
dropout/PartitionedCallæ
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pixel_attention2d_1443536pixel_attention2d_1443538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_14432322+
)pixel_attention2d/StatefulPartitionedCall
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_14432642
lambda/PartitionedCalló
flatten_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_14432832
flatten_1/PartitionedCallë
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_14432972
flatten/PartitionedCall¯
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1443544dense_1_1443546*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14433162!
dense_1/StatefulPartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1443549dense_1443551*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_14433432
dense/StatefulPartitionedCall¹
 b_output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0b_output_1443554b_output_1443556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_b_output_layer_call_and_return_conditional_losses_14433702"
 b_output/StatefulPartitionedCall¼
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_1443559mc_output_1443561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_mc_output_layer_call_and_return_conditional_losses_14433972#
!mc_output/StatefulPartitionedCallÓ
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÖ

Identity_1Identity)b_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2D
 b_output/StatefulPartitionedCall b_output/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
Á
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_1443876
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/BiasAdd_1/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d/Conv2D_1/ReadVariableOpª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAddp
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoidg
MulMulxconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Mul®
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d/Conv2D_1/ReadVariableOpÀ
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D_1¥
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp¬
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAdd_1÷
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
Î
c
D__inference_dropout_layer_call_and_return_conditional_losses_1443845

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ä?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÈ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬1>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ã

+__inference_mc_output_layer_call_fn_1444009

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_mc_output_layer_call_and_return_conditional_losses_14433972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¤
'__inference_model_layer_call_fn_1443525
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_14434962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

£
'__inference_model_layer_call_fn_1443813

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_14435662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
D
(__inference_lambda_layer_call_fn_1443927

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_14432642
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


__inference_call_1443901
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/BiasAdd_1/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d/Conv2D_1/ReadVariableOpª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv2d/Conv2D/ReadVariableOp´
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAddp
SigmoidSigmoidconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
Sigmoidg
MulMulxconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Mul®
conv2d/Conv2D_1/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d/Conv2D_1/ReadVariableOpÀ
conv2d/Conv2D_1Conv2DMul:z:0&conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d/Conv2D_1¥
conv2d/BiasAdd_1/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d/BiasAdd_1/ReadVariableOp¬
conv2d/BiasAdd_1BiasAddconv2d/Conv2D_1:output:0'conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d/BiasAdd_1÷
IdentityIdentityconv2d/BiasAdd_1:output:0^conv2d/BiasAdd/ReadVariableOp ^conv2d/BiasAdd_1/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^conv2d/Conv2D_1/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2B
conv2d/BiasAdd_1/ReadVariableOpconv2d/BiasAdd_1/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
conv2d/Conv2D_1/ReadVariableOpconv2d/Conv2D_1/ReadVariableOp:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex


3__inference_pixel_attention2d_layer_call_fn_1443885
x
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_14432322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@

_user_specified_namex
¶1
÷
B__inference_model_layer_call_and_return_conditional_losses_1443415
input_1
conv0_1443181
conv0_1443183
pixel_attention2d_1443243
pixel_attention2d_1443245
dense_1_1443327
dense_1_1443329
dense_1443354
dense_1443356
b_output_1443381
b_output_1443383
mc_output_1443408
mc_output_1443410
identity

identity_1¢ b_output/StatefulPartitionedCall¢conv0/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!mc_output/StatefulPartitionedCall¢)pixel_attention2d/StatefulPartitionedCall
conv0/StatefulPartitionedCallStatefulPartitionedCallinput_1conv0_1443181conv0_1443183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_14431702
conv0/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_14431982!
dropout/StatefulPartitionedCallî
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0pixel_attention2d_1443243pixel_attention2d_1443245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_14432322+
)pixel_attention2d/StatefulPartitionedCall
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_14432562
lambda/PartitionedCallû
flatten_1/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_14432832
flatten_1/PartitionedCallë
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_14432972
flatten/PartitionedCall¯
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1443327dense_1_1443329*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14433162!
dense_1/StatefulPartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1443354dense_1443356*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_14433432
dense/StatefulPartitionedCall¹
 b_output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0b_output_1443381b_output_1443383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_b_output_layer_call_and_return_conditional_losses_14433702"
 b_output/StatefulPartitionedCall¼
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_1443408mc_output_1443410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_mc_output_layer_call_and_return_conditional_losses_14433972#
!mc_output/StatefulPartitionedCallõ
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityø

Identity_1Identity)b_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2D
 b_output/StatefulPartitionedCall b_output/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
à

*__inference_b_output_layer_call_fn_1444029

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_b_output_layer_call_and_return_conditional_losses_14433702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä/
Õ
B__inference_model_layer_call_and_return_conditional_losses_1443454
input_1
conv0_1443418
conv0_1443420
pixel_attention2d_1443424
pixel_attention2d_1443426
dense_1_1443432
dense_1_1443434
dense_1443437
dense_1443439
b_output_1443442
b_output_1443444
mc_output_1443447
mc_output_1443449
identity

identity_1¢ b_output/StatefulPartitionedCall¢conv0/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!mc_output/StatefulPartitionedCall¢)pixel_attention2d/StatefulPartitionedCall
conv0/StatefulPartitionedCallStatefulPartitionedCallinput_1conv0_1443418conv0_1443420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_14431702
conv0/StatefulPartitionedCallú
dropout/PartitionedCallPartitionedCall&conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_14432032
dropout/PartitionedCallæ
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0pixel_attention2d_1443424pixel_attention2d_1443426*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_14432322+
)pixel_attention2d/StatefulPartitionedCall
lambda/PartitionedCallPartitionedCall2pixel_attention2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lambda_layer_call_and_return_conditional_losses_14432642
lambda/PartitionedCalló
flatten_1/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_14432832
flatten_1/PartitionedCallë
flatten/PartitionedCallPartitionedCalllambda/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_14432972
flatten/PartitionedCall¯
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_1443432dense_1_1443434*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14433162!
dense_1/StatefulPartitionedCall£
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1443437dense_1443439*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_14433432
dense/StatefulPartitionedCall¹
 b_output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0b_output_1443442b_output_1443444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_b_output_layer_call_and_return_conditional_losses_14433702"
 b_output/StatefulPartitionedCall¼
!mc_output/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0mc_output_1443447mc_output_1443449*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_mc_output_layer_call_and_return_conditional_losses_14433972#
!mc_output/StatefulPartitionedCallÓ
IdentityIdentity*mc_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÖ

Identity_1Identity)b_output/StatefulPartitionedCall:output:0!^b_output/StatefulPartitionedCall^conv0/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^mc_output/StatefulPartitionedCall*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2D
 b_output/StatefulPartitionedCall b_output/StatefulPartitionedCall2>
conv0/StatefulPartitionedCallconv0/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!mc_output/StatefulPartitionedCall!mc_output/StatefulPartitionedCall2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ä
_
C__inference_lambda_layer_call_and_return_conditional_losses_1443256

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1443944

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
b
)__inference_dropout_layer_call_fn_1443855

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_14431982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï^
Í
 __inference__traced_save_1444194
file_prefix+
'savev2_conv0_kernel_read_readvariableop)
%savev2_conv0_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop/
+savev2_mc_output_kernel_read_readvariableop-
)savev2_mc_output_bias_read_readvariableop.
*savev2_b_output_kernel_read_readvariableop,
(savev2_b_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop2
.savev2_adam_conv0_kernel_m_read_readvariableop0
,savev2_adam_conv0_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop6
2savev2_adam_mc_output_kernel_m_read_readvariableop4
0savev2_adam_mc_output_bias_m_read_readvariableop5
1savev2_adam_b_output_kernel_m_read_readvariableop3
/savev2_adam_b_output_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop2
.savev2_adam_conv0_kernel_v_read_readvariableop0
,savev2_adam_conv0_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop6
2savev2_adam_mc_output_kernel_v_read_readvariableop4
0savev2_adam_mc_output_bias_v_read_readvariableop5
1savev2_adam_b_output_kernel_v_read_readvariableop3
/savev2_adam_b_output_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesè
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv0_kernel_read_readvariableop%savev2_conv0_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop+savev2_mc_output_kernel_read_readvariableop)savev2_mc_output_bias_read_readvariableop*savev2_b_output_kernel_read_readvariableop(savev2_b_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop.savev2_adam_conv0_kernel_m_read_readvariableop,savev2_adam_conv0_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop2savev2_adam_mc_output_kernel_m_read_readvariableop0savev2_adam_mc_output_bias_m_read_readvariableop1savev2_adam_b_output_kernel_m_read_readvariableop/savev2_adam_b_output_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop.savev2_adam_conv0_kernel_v_read_readvariableop,savev2_adam_conv0_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop2savev2_adam_mc_output_kernel_v_read_readvariableop0savev2_adam_mc_output_bias_v_read_readvariableop1savev2_adam_b_output_kernel_v_read_readvariableop/savev2_adam_b_output_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapes
ÿ: :@:@:	y::
À<::	::	:: : : : : :@@:@: : : : : : :@:@:	y::
À<::	::	::@@:@:@:@:	y::
À<::	::	::@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	y:!

_output_shapes	
::&"
 
_output_shapes
:
À<:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%	!

_output_shapes
:	: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	y:!

_output_shapes	
::&"
 
_output_shapes
:
À<:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::% !

_output_shapes
:	: !

_output_shapes
::,"(
&
_output_shapes
:@@: #

_output_shapes
:@:,$(
&
_output_shapes
:@: %

_output_shapes
:@:%&!

_output_shapes
:	y:!'

_output_shapes	
::&("
 
_output_shapes
:
À<:!)

_output_shapes	
::%*!

_output_shapes
:	: +

_output_shapes
::%,!

_output_shapes
:	: -

_output_shapes
::,.(
&
_output_shapes
:@@: /

_output_shapes
:@:0

_output_shapes
: 
ÄK
Ä
"__inference__wrapped_model_1443155
input_1.
*model_conv0_conv2d_readvariableop_resource/
+model_conv0_biasadd_readvariableop_resource#
model_pixel_attention2d_1443112#
model_pixel_attention2d_14431140
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource1
-model_b_output_matmul_readvariableop_resource2
.model_b_output_biasadd_readvariableop_resource2
.model_mc_output_matmul_readvariableop_resource3
/model_mc_output_biasadd_readvariableop_resource
identity

identity_1¢%model/b_output/BiasAdd/ReadVariableOp¢$model/b_output/MatMul/ReadVariableOp¢"model/conv0/BiasAdd/ReadVariableOp¢!model/conv0/Conv2D/ReadVariableOp¢"model/dense/BiasAdd/ReadVariableOp¢!model/dense/MatMul/ReadVariableOp¢$model/dense_1/BiasAdd/ReadVariableOp¢#model/dense_1/MatMul/ReadVariableOp¢&model/mc_output/BiasAdd/ReadVariableOp¢%model/mc_output/MatMul/ReadVariableOp¢/model/pixel_attention2d/StatefulPartitionedCall¹
!model/conv0/Conv2D/ReadVariableOpReadVariableOp*model_conv0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02#
!model/conv0/Conv2D/ReadVariableOpÉ
model/conv0/Conv2DConv2Dinput_1)model/conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
model/conv0/Conv2D°
"model/conv0/BiasAdd/ReadVariableOpReadVariableOp+model_conv0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"model/conv0/BiasAdd/ReadVariableOp¸
model/conv0/BiasAddBiasAddmodel/conv0/Conv2D:output:0*model/conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/conv0/BiasAdd
model/conv0/ReluRelumodel/conv0/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/conv0/Relu
model/dropout/IdentityIdentitymodel/conv0/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
model/dropout/IdentityÇ
/model/pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCallmodel/dropout/Identity:output:0model_pixel_attention2d_1443112model_pixel_attention2d_1443114*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_144311121
/model/pixel_attention2d/StatefulPartitionedCall
#model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#model/lambda/Mean/reduction_indicesÌ
model/lambda/MeanMean8model/pixel_attention2d/StatefulPartitionedCall:output:0,model/lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/lambda/Mean
model/lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
model/lambda/ExpandDims/dim¼
model/lambda/ExpandDims
ExpandDimsmodel/lambda/Mean:output:0$model/lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/lambda/ExpandDims
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
model/flatten_1/Const±
model/flatten_1/ReshapeReshapemodel/dropout/Identity:output:0model/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2
model/flatten_1/Reshape{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿy   2
model/flatten/Const«
model/flatten/ReshapeReshape model/lambda/ExpandDims:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2
model/flatten/Reshape¹
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
À<*
dtype02%
#model/dense_1/MatMul/ReadVariableOp¸
model/dense_1/MatMulMatMul model/flatten_1/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_1/MatMul·
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOpº
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_1/BiasAdd
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense_1/Relu²
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	y*
dtype02#
!model/dense/MatMul/ReadVariableOp°
model/dense/MatMulMatMulmodel/flatten/Reshape:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/MatMul±
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp²
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/BiasAdd}
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/dense/Relu»
$model/b_output/MatMul/ReadVariableOpReadVariableOp-model_b_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02&
$model/b_output/MatMul/ReadVariableOpº
model/b_output/MatMulMatMul model/dense_1/Relu:activations:0,model/b_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/b_output/MatMul¹
%model/b_output/BiasAdd/ReadVariableOpReadVariableOp.model_b_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/b_output/BiasAdd/ReadVariableOp½
model/b_output/BiasAddBiasAddmodel/b_output/MatMul:product:0-model/b_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/b_output/BiasAdd
model/b_output/SigmoidSigmoidmodel/b_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/b_output/Sigmoid¾
%model/mc_output/MatMul/ReadVariableOpReadVariableOp.model_mc_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%model/mc_output/MatMul/ReadVariableOp»
model/mc_output/MatMulMatMulmodel/dense/Relu:activations:0-model/mc_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/mc_output/MatMul¼
&model/mc_output/BiasAdd/ReadVariableOpReadVariableOp/model_mc_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model/mc_output/BiasAdd/ReadVariableOpÁ
model/mc_output/BiasAddBiasAdd model/mc_output/MatMul:product:0.model/mc_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/mc_output/BiasAdd
model/mc_output/SoftmaxSoftmax model/mc_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model/mc_output/Softmax
IdentityIdentitymodel/b_output/Sigmoid:y:0&^model/b_output/BiasAdd/ReadVariableOp%^model/b_output/MatMul/ReadVariableOp#^model/conv0/BiasAdd/ReadVariableOp"^model/conv0/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp'^model/mc_output/BiasAdd/ReadVariableOp&^model/mc_output/MatMul/ReadVariableOp0^model/pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityª

Identity_1Identity!model/mc_output/Softmax:softmax:0&^model/b_output/BiasAdd/ReadVariableOp%^model/b_output/MatMul/ReadVariableOp#^model/conv0/BiasAdd/ReadVariableOp"^model/conv0/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp'^model/mc_output/BiasAdd/ReadVariableOp&^model/mc_output/MatMul/ReadVariableOp0^model/pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2N
%model/b_output/BiasAdd/ReadVariableOp%model/b_output/BiasAdd/ReadVariableOp2L
$model/b_output/MatMul/ReadVariableOp$model/b_output/MatMul/ReadVariableOp2H
"model/conv0/BiasAdd/ReadVariableOp"model/conv0/BiasAdd/ReadVariableOp2F
!model/conv0/Conv2D/ReadVariableOp!model/conv0/Conv2D/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2P
&model/mc_output/BiasAdd/ReadVariableOp&model/mc_output/BiasAdd/ReadVariableOp2N
%model/mc_output/MatMul/ReadVariableOp%model/mc_output/MatMul/ReadVariableOp2b
/model/pixel_attention2d/StatefulPartitionedCall/model/pixel_attention2d/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
º
`
D__inference_flatten_layer_call_and_return_conditional_losses_1443297

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿy   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
C
Ù
B__inference_model_layer_call_and_return_conditional_losses_1443751

inputs(
$conv0_conv2d_readvariableop_resource)
%conv0_biasadd_readvariableop_resource
pixel_attention2d_1443708
pixel_attention2d_1443710*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource+
'b_output_matmul_readvariableop_resource,
(b_output_biasadd_readvariableop_resource,
(mc_output_matmul_readvariableop_resource-
)mc_output_biasadd_readvariableop_resource
identity

identity_1¢b_output/BiasAdd/ReadVariableOp¢b_output/MatMul/ReadVariableOp¢conv0/BiasAdd/ReadVariableOp¢conv0/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢ mc_output/BiasAdd/ReadVariableOp¢mc_output/MatMul/ReadVariableOp¢)pixel_attention2d/StatefulPartitionedCall§
conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv0/Conv2D/ReadVariableOp¶
conv0/Conv2DConv2Dinputs#conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv0/Conv2D
conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv0/BiasAdd/ReadVariableOp 
conv0/BiasAddBiasAddconv0/Conv2D:output:0$conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv0/BiasAddr

conv0/ReluReluconv0/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

conv0/Relu
dropout/IdentityIdentityconv0/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Identity©
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCalldropout/Identity:output:0pixel_attention2d_1443708pixel_attention2d_1443710*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_14431112+
)pixel_attention2d/StatefulPartitionedCall
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lambda/Mean/reduction_indices´
lambda/MeanMean2pixel_attention2d/StatefulPartitionedCall:output:0&lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lambda/Meany
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lambda/ExpandDims/dim¤
lambda/ExpandDims
ExpandDimslambda/Mean:output:0lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lambda/ExpandDimss
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_1/Const
flatten_1/ReshapeReshapedropout/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿy   2
flatten/Const
flatten/ReshapeReshapelambda/ExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2
flatten/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
À<*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	y*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu©
b_output/MatMul/ReadVariableOpReadVariableOp'b_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
b_output/MatMul/ReadVariableOp¢
b_output/MatMulMatMuldense_1/Relu:activations:0&b_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
b_output/MatMul§
b_output/BiasAdd/ReadVariableOpReadVariableOp(b_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
b_output/BiasAdd/ReadVariableOp¥
b_output/BiasAddBiasAddb_output/MatMul:product:0'b_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
b_output/BiasAdd|
b_output/SigmoidSigmoidb_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
b_output/Sigmoid¬
mc_output/MatMul/ReadVariableOpReadVariableOp(mc_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
mc_output/MatMul/ReadVariableOp£
mc_output/MatMulMatMuldense/Relu:activations:0'mc_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_output/MatMulª
 mc_output/BiasAdd/ReadVariableOpReadVariableOp)mc_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 mc_output/BiasAdd/ReadVariableOp©
mc_output/BiasAddBiasAddmc_output/MatMul:product:0(mc_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_output/BiasAdd
mc_output/SoftmaxSoftmaxmc_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_output/SoftmaxÞ
IdentityIdentitymc_output/Softmax:softmax:0 ^b_output/BiasAdd/ReadVariableOp^b_output/MatMul/ReadVariableOp^conv0/BiasAdd/ReadVariableOp^conv0/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp!^mc_output/BiasAdd/ReadVariableOp ^mc_output/MatMul/ReadVariableOp*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÛ

Identity_1Identityb_output/Sigmoid:y:0 ^b_output/BiasAdd/ReadVariableOp^b_output/MatMul/ReadVariableOp^conv0/BiasAdd/ReadVariableOp^conv0/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp!^mc_output/BiasAdd/ReadVariableOp ^mc_output/MatMul/ReadVariableOp*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2B
b_output/BiasAdd/ReadVariableOpb_output/BiasAdd/ReadVariableOp2@
b_output/MatMul/ReadVariableOpb_output/MatMul/ReadVariableOp2<
conv0/BiasAdd/ReadVariableOpconv0/BiasAdd/ReadVariableOp2:
conv0/Conv2D/ReadVariableOpconv0/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2D
 mc_output/BiasAdd/ReadVariableOp mc_output/BiasAdd/ReadVariableOp2B
mc_output/MatMul/ReadVariableOpmc_output/MatMul/ReadVariableOp2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯L
Ù
B__inference_model_layer_call_and_return_conditional_losses_1443697

inputs(
$conv0_conv2d_readvariableop_resource)
%conv0_biasadd_readvariableop_resource
pixel_attention2d_1443654
pixel_attention2d_1443656*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource+
'b_output_matmul_readvariableop_resource,
(b_output_biasadd_readvariableop_resource,
(mc_output_matmul_readvariableop_resource-
)mc_output_biasadd_readvariableop_resource
identity

identity_1¢b_output/BiasAdd/ReadVariableOp¢b_output/MatMul/ReadVariableOp¢conv0/BiasAdd/ReadVariableOp¢conv0/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp¢ mc_output/BiasAdd/ReadVariableOp¢mc_output/MatMul/ReadVariableOp¢)pixel_attention2d/StatefulPartitionedCall§
conv0/Conv2D/ReadVariableOpReadVariableOp$conv0_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv0/Conv2D/ReadVariableOp¶
conv0/Conv2DConv2Dinputs#conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv0/Conv2D
conv0/BiasAdd/ReadVariableOpReadVariableOp%conv0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv0/BiasAdd/ReadVariableOp 
conv0/BiasAddBiasAddconv0/Conv2D:output:0$conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv0/BiasAddr

conv0/ReluReluconv0/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

conv0/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ä?2
dropout/dropout/Const¥
dropout/dropout/MulMulconv0/Relu:activations:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeconv0/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeà
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed2.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬1>2 
dropout/dropout/GreaterEqual/yæ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Cast¢
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/dropout/Mul_1©
)pixel_attention2d/StatefulPartitionedCallStatefulPartitionedCalldropout/dropout/Mul_1:z:0pixel_attention2d_1443654pixel_attention2d_1443656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_call_14431112+
)pixel_attention2d/StatefulPartitionedCall
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lambda/Mean/reduction_indices´
lambda/MeanMean2pixel_attention2d/StatefulPartitionedCall:output:0&lambda/Mean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lambda/Meany
lambda/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lambda/ExpandDims/dim¤
lambda/ExpandDims
ExpandDimslambda/Mean:output:0lambda/ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lambda/ExpandDimss
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
flatten_1/Const
flatten_1/ReshapeReshapedropout/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2
flatten_1/Reshapeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿy   2
flatten/Const
flatten/ReshapeReshapelambda/ExpandDims:output:0flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2
flatten/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
À<*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	y*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu©
b_output/MatMul/ReadVariableOpReadVariableOp'b_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
b_output/MatMul/ReadVariableOp¢
b_output/MatMulMatMuldense_1/Relu:activations:0&b_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
b_output/MatMul§
b_output/BiasAdd/ReadVariableOpReadVariableOp(b_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
b_output/BiasAdd/ReadVariableOp¥
b_output/BiasAddBiasAddb_output/MatMul:product:0'b_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
b_output/BiasAdd|
b_output/SigmoidSigmoidb_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
b_output/Sigmoid¬
mc_output/MatMul/ReadVariableOpReadVariableOp(mc_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02!
mc_output/MatMul/ReadVariableOp£
mc_output/MatMulMatMuldense/Relu:activations:0'mc_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_output/MatMulª
 mc_output/BiasAdd/ReadVariableOpReadVariableOp)mc_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 mc_output/BiasAdd/ReadVariableOp©
mc_output/BiasAddBiasAddmc_output/MatMul:product:0(mc_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_output/BiasAdd
mc_output/SoftmaxSoftmaxmc_output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mc_output/SoftmaxÞ
IdentityIdentitymc_output/Softmax:softmax:0 ^b_output/BiasAdd/ReadVariableOp^b_output/MatMul/ReadVariableOp^conv0/BiasAdd/ReadVariableOp^conv0/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp!^mc_output/BiasAdd/ReadVariableOp ^mc_output/MatMul/ReadVariableOp*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÛ

Identity_1Identityb_output/Sigmoid:y:0 ^b_output/BiasAdd/ReadVariableOp^b_output/MatMul/ReadVariableOp^conv0/BiasAdd/ReadVariableOp^conv0/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp!^mc_output/BiasAdd/ReadVariableOp ^mc_output/MatMul/ReadVariableOp*^pixel_attention2d/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::2B
b_output/BiasAdd/ReadVariableOpb_output/BiasAdd/ReadVariableOp2@
b_output/MatMul/ReadVariableOpb_output/MatMul/ReadVariableOp2<
conv0/BiasAdd/ReadVariableOpconv0/BiasAdd/ReadVariableOp2:
conv0/Conv2D/ReadVariableOpconv0/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2D
 mc_output/BiasAdd/ReadVariableOp mc_output/BiasAdd/ReadVariableOp2B
mc_output/MatMul/ReadVariableOpmc_output/MatMul/ReadVariableOp2V
)pixel_attention2d/StatefulPartitionedCall)pixel_attention2d/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

Û
B__inference_conv0_layer_call_and_return_conditional_losses_1443170

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_1443283

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
c
D__inference_dropout_layer_call_and_return_conditional_losses_1443198

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Ä?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÈ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬1>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
à
~
)__inference_dense_1_layer_call_fn_1443989

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_14433162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ<::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<
 
_user_specified_nameinputs
¢
E
)__inference_flatten_layer_call_fn_1443938

inputs
identityÂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_14432972
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
|
'__inference_conv0_layer_call_fn_1443833

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv0_layer_call_and_return_conditional_losses_14431702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
Ý
D__inference_dense_1_layer_call_and_return_conditional_losses_1443980

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
À<*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀ<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<
 
_user_specified_nameinputs
ô	
Þ
E__inference_b_output_layer_call_and_return_conditional_losses_1443370

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
E
)__inference_dropout_layer_call_fn_1443860

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_14432032
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ç
b
D__inference_dropout_layer_call_and_return_conditional_losses_1443850

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

£
'__inference_model_layer_call_fn_1443782

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_14434962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä
_
C__inference_lambda_layer_call_and_return_conditional_losses_1443909

inputs
identity{
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
Mean/reduction_indicess
MeanMeaninputsMean/reduction_indices:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Meank
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
ExpandDims/dim

ExpandDims
ExpandDimsMean:output:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimso
IdentityIdentityExpandDims:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
|
'__inference_dense_layer_call_fn_1443969

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_14433432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿy::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 
_user_specified_nameinputs
ò	
Û
B__inference_dense_layer_call_and_return_conditional_losses_1443343

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	y*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿy::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
 
_user_specified_nameinputs
¨
G
+__inference_flatten_1_layer_call_fn_1443949

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_14432832
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ<2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ò
serving_defaultÞ
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
b_output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ=
	mc_output0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÂÕ
ÀR
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
	optimizer
loss
trainable_variables
	variables
regularization_losses
	keras_api

signatures
´_default_save_signature
µ__call__
+¶&call_and_return_all_conditional_losses"¼N
_tf_keras_network N{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv0", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1729533112927278, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv0", 0, 0, {}]]]}, {"class_name": "PixelAttention2D", "config": {"name": "pixel_attention2d", "trainable": true, "dtype": "float32", "Att_filters": 64}, "name": "pixel_attention2d", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxIAAAB0AHQBfABkA2QCjQJkBIMCUwApBU7pAQAAACkB2gRh\neGlz6f////9yAwAAACkC2gtleHBhbmRfZGltc9oLcmVkdWNlX21lYW4pAdoBeKkAcgcAAAD6LC9o\nb21lL29tZW4tZ2l1c3kvQXR0ZW50aW9uL1hBSS9UcmFpbk1vZGVsLnB52gg8bGFtYmRhPvMAAADz\nAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "XAI.TrainModel", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["pixel_attention2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mc_output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mc_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "b_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "b_output", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["mc_output", 0, 0], ["b_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv0", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1729533112927278, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["conv0", 0, 0, {}]]]}, {"class_name": "PixelAttention2D", "config": {"name": "pixel_attention2d", "trainable": true, "dtype": "float32", "Att_filters": 64}, "name": "pixel_attention2d", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxIAAAB0AHQBfABkA2QCjQJkBIMCUwApBU7pAQAAACkB2gRh\neGlz6f////9yAwAAACkC2gtleHBhbmRfZGltc9oLcmVkdWNlX21lYW4pAdoBeKkAcgcAAAD6LC9o\nb21lL29tZW4tZ2l1c3kvQXR0ZW50aW9uL1hBSS9UcmFpbk1vZGVsLnB52gg8bGFtYmRhPvMAAADz\nAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "XAI.TrainModel", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["pixel_attention2d", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["lambda", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mc_output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mc_output", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "b_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "b_output", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["mc_output", 0, 0], ["b_output", 0, 0]]}}, "training_config": {"loss": {"mc_output": "categorical_crossentropy", "b_output": "binary_crossentropy"}, "metrics": null, "weighted_metrics": null, "loss_weights": [0.805981126889679, 0.19401887311032096], "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005360077484510839, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ù"ö
_tf_keras_input_layerÖ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12, 12, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
í	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Conv2D", "name": "conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv0", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 1]}}
ò
trainable_variables
	variables
regularization_losses
	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"á
_tf_keras_layerÇ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1729533112927278, "noise_shape": null, "seed": null}}
ú
	conv1
trainable_variables
	variables
 regularization_losses
!	keras_api
»__call__
+¼&call_and_return_all_conditional_losses
	½call"Ó
_tf_keras_layer¹{"class_name": "PixelAttention2D", "name": "pixel_attention2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "pixel_attention2d", "trainable": true, "dtype": "float32", "Att_filters": 64}}
ÿ
"trainable_variables
#	variables
$regularization_losses
%	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAUAAABTAAAAcxIAAAB0AHQBfABkA2QCjQJkBIMCUwApBU7pAQAAACkB2gRh\neGlz6f////9yAwAAACkC2gtleHBhbmRfZGltc9oLcmVkdWNlX21lYW4pAdoBeKkAcgcAAAD6LC9o\nb21lL29tZW4tZ2l1c3kvQXR0ZW50aW9uL1hBSS9UcmFpbk1vZGVsLnB52gg8bGFtYmRhPvMAAADz\nAAAAAA==\n", null, null]}, "function_type": "lambda", "module": "XAI.TrainModel", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
ä
&trainable_variables
'	variables
(regularization_losses
)	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
è
*trainable_variables
+	variables
,regularization_losses
-	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ñ

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 121}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 121]}}
÷

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
Æ__call__
+Ç&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 7744}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7744]}}
ú

:kernel
;bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
È__call__
+É&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Dense", "name": "mc_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "mc_output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ø

@kernel
Abias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "b_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "b_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Ã
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratemm.m/m4m 5m¡:m¢;m£@m¤Am¥Km¦Lm§v¨v©.vª/v«4v¬5v­:v®;v¯@v°Av±Kv²Lv³"
	optimizer
 "
trackable_dict_wrapper
v
0
1
K2
L3
.4
/5
46
57
:8
;9
@10
A11"
trackable_list_wrapper
v
0
1
K2
L3
.4
/5
46
57
:8
;9
@10
A11"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
trainable_variables
Mlayer_regularization_losses
	variables
Nlayer_metrics
regularization_losses

Olayers
Pnon_trainable_variables
Qmetrics
µ__call__
´_default_save_signature
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
-
Ìserving_default"
signature_map
&:$@2conv0/kernel
:@2
conv0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Rlayer_regularization_losses
trainable_variables
	variables
Slayer_metrics
regularization_losses

Tlayers
Unon_trainable_variables
Vmetrics
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Wlayer_regularization_losses
trainable_variables
	variables
Xlayer_metrics
regularization_losses

Ylayers
Znon_trainable_variables
[metrics
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
ó	

Kkernel
Lbias
\trainable_variables
]	variables
^regularization_losses
_	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 11, 11, 64]}}
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
`layer_regularization_losses
trainable_variables
	variables
alayer_metrics
 regularization_losses

blayers
cnon_trainable_variables
dmetrics
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
elayer_regularization_losses
"trainable_variables
#	variables
flayer_metrics
$regularization_losses

glayers
hnon_trainable_variables
imetrics
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
jlayer_regularization_losses
&trainable_variables
'	variables
klayer_metrics
(regularization_losses

llayers
mnon_trainable_variables
nmetrics
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
olayer_regularization_losses
*trainable_variables
+	variables
player_metrics
,regularization_losses

qlayers
rnon_trainable_variables
smetrics
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
:	y2dense/kernel
:2
dense/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
tlayer_regularization_losses
0trainable_variables
1	variables
ulayer_metrics
2regularization_losses

vlayers
wnon_trainable_variables
xmetrics
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
_generic_user_object
": 
À<2dense_1/kernel
:2dense_1/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
°
ylayer_regularization_losses
6trainable_variables
7	variables
zlayer_metrics
8regularization_losses

{layers
|non_trainable_variables
}metrics
Æ__call__
+Ç&call_and_return_all_conditional_losses
'Ç"call_and_return_conditional_losses"
_generic_user_object
#:!	2mc_output/kernel
:2mc_output/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
³
~layer_regularization_losses
<trainable_variables
=	variables
layer_metrics
>regularization_losses
layers
non_trainable_variables
metrics
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
": 	2b_output/kernel
:2b_output/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
Btrainable_variables
C	variables
layer_metrics
Dregularization_losses
layers
non_trainable_variables
metrics
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%@@2conv2d/kernel
:@2conv2d/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
8
0
1
2"
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
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layer_regularization_losses
\trainable_variables
]	variables
layer_metrics
^regularization_losses
layers
non_trainable_variables
metrics
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
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
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ó

total

count
	variables
	keras_api"
_tf_keras_metric~{"class_name": "Mean", "name": "mc_output_loss", "dtype": "float32", "config": {"name": "mc_output_loss", "dtype": "float32"}}
Ñ

total

count
	variables
	keras_api"
_tf_keras_metric|{"class_name": "Mean", "name": "b_output_loss", "dtype": "float32", "config": {"name": "b_output_loss", "dtype": "float32"}}
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
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
+:)@2Adam/conv0/kernel/m
:@2Adam/conv0/bias/m
$:"	y2Adam/dense/kernel/m
:2Adam/dense/bias/m
':%
À<2Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
(:&	2Adam/mc_output/kernel/m
!:2Adam/mc_output/bias/m
':%	2Adam/b_output/kernel/m
 :2Adam/b_output/bias/m
,:*@@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
+:)@2Adam/conv0/kernel/v
:@2Adam/conv0/bias/v
$:"	y2Adam/dense/kernel/v
:2Adam/dense/bias/v
':%
À<2Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
(:&	2Adam/mc_output/kernel/v
!:2Adam/mc_output/bias/v
':%	2Adam/b_output/kernel/v
 :2Adam/b_output/bias/v
,:*@@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
è2å
"__inference__wrapped_model_1443155¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ê2ç
'__inference_model_layer_call_fn_1443813
'__inference_model_layer_call_fn_1443782
'__inference_model_layer_call_fn_1443525
'__inference_model_layer_call_fn_1443595À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_layer_call_and_return_conditional_losses_1443751
B__inference_model_layer_call_and_return_conditional_losses_1443454
B__inference_model_layer_call_and_return_conditional_losses_1443697
B__inference_model_layer_call_and_return_conditional_losses_1443415À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_conv0_layer_call_fn_1443833¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_conv0_layer_call_and_return_conditional_losses_1443824¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_dropout_layer_call_fn_1443860
)__inference_dropout_layer_call_fn_1443855´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_layer_call_and_return_conditional_losses_1443845
D__inference_dropout_layer_call_and_return_conditional_losses_1443850´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
3__inference_pixel_attention2d_layer_call_fn_1443885
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_1443876
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
½2º
__inference_call_1443901
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_lambda_layer_call_fn_1443922
(__inference_lambda_layer_call_fn_1443927À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
C__inference_lambda_layer_call_and_return_conditional_losses_1443909
C__inference_lambda_layer_call_and_return_conditional_losses_1443917À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_flatten_layer_call_fn_1443938¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_1443933¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_flatten_1_layer_call_fn_1443949¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_flatten_1_layer_call_and_return_conditional_losses_1443944¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_layer_call_fn_1443969¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_layer_call_and_return_conditional_losses_1443960¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_1_layer_call_fn_1443989¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_1_layer_call_and_return_conditional_losses_1443980¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_mc_output_layer_call_fn_1444009¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_mc_output_layer_call_and_return_conditional_losses_1444000¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_b_output_layer_call_fn_1444029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_b_output_layer_call_and_return_conditional_losses_1444020¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
%__inference_signature_wrapper_1443636input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ö
"__inference__wrapped_model_1443155¯KL45./@A:;8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "eªb
.
b_output"
b_outputÿÿÿÿÿÿÿÿÿ
0
	mc_output# 
	mc_outputÿÿÿÿÿÿÿÿÿ¦
E__inference_b_output_layer_call_and_return_conditional_losses_1444020]@A0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
*__inference_b_output_layer_call_fn_1444029P@A0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿv
__inference_call_1443901ZKL2¢/
(¢%
# 
xÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@²
B__inference_conv0_layer_call_and_return_conditional_losses_1443824l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
'__inference_conv0_layer_call_fn_1443833_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@¦
D__inference_dense_1_layer_call_and_return_conditional_losses_1443980^450¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ<
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_1_layer_call_fn_1443989Q450¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀ<
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_layer_call_and_return_conditional_losses_1443960].//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿy
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_dense_layer_call_fn_1443969P.//¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿy
ª "ÿÿÿÿÿÿÿÿÿ´
D__inference_dropout_layer_call_and_return_conditional_losses_1443845l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ´
D__inference_dropout_layer_call_and_return_conditional_losses_1443850l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_dropout_layer_call_fn_1443855_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
)__inference_dropout_layer_call_fn_1443860_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@«
F__inference_flatten_1_layer_call_and_return_conditional_losses_1443944a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀ<
 
+__inference_flatten_1_layer_call_fn_1443949T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀ<¨
D__inference_flatten_layer_call_and_return_conditional_losses_1443933`7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿy
 
)__inference_flatten_layer_call_fn_1443938S7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿy·
C__inference_lambda_layer_call_and_return_conditional_losses_1443909p?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@

 
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_lambda_layer_call_and_return_conditional_losses_1443917p?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@

 
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_lambda_layer_call_fn_1443922c?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@

 
p
ª " ÿÿÿÿÿÿÿÿÿ
(__inference_lambda_layer_call_fn_1443927c?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ@

 
p 
ª " ÿÿÿÿÿÿÿÿÿ§
F__inference_mc_output_layer_call_and_return_conditional_losses_1444000]:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_mc_output_layer_call_fn_1444009P:;0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿä
B__inference_model_layer_call_and_return_conditional_losses_1443415KL45./@A:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ä
B__inference_model_layer_call_and_return_conditional_losses_1443454KL45./@A:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ã
B__inference_model_layer_call_and_return_conditional_losses_1443697KL45./@A:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ã
B__inference_model_layer_call_and_return_conditional_losses_1443751KL45./@A:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "K¢H
A>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 »
'__inference_model_layer_call_fn_1443525KL45./@A:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ»
'__inference_model_layer_call_fn_1443595KL45./@A:;@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿº
'__inference_model_layer_call_fn_1443782KL45./@A:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿº
'__inference_model_layer_call_fn_1443813KL45./@A:;?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "=:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ¹
N__inference_pixel_attention2d_layer_call_and_return_conditional_losses_1443876gKL2¢/
(¢%
# 
xÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
3__inference_pixel_attention2d_layer_call_fn_1443885ZKL2¢/
(¢%
# 
xÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ä
%__inference_signature_wrapper_1443636ºKL45./@A:;C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ"eªb
.
b_output"
b_outputÿÿÿÿÿÿÿÿÿ
0
	mc_output# 
	mc_outputÿÿÿÿÿÿÿÿÿ