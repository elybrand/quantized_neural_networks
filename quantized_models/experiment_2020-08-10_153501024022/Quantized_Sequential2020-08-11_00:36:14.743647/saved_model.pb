УП%
™э
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8иц
Д
conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_36/kernel
}
$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*&
_output_shapes
: *
dtype0
t
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_36/bias
m
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes
: *
dtype0
Р
batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_37/gamma
Й
0batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_37/gamma*
_output_shapes
: *
dtype0
О
batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_37/beta
З
/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_37/beta*
_output_shapes
: *
dtype0
Ь
"batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_37/moving_mean
Х
6batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_37/moving_mean*
_output_shapes
: *
dtype0
§
&batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_37/moving_variance
Э
:batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_37/moving_variance*
_output_shapes
: *
dtype0
И
conv2d_1_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *#
shared_nameconv2d_1_31/kernel
Б
&conv2d_1_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_1_31/kernel*&
_output_shapes
:  *
dtype0
x
conv2d_1_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_1_31/bias
q
$conv2d_1_31/bias/Read/ReadVariableOpReadVariableOpconv2d_1_31/bias*
_output_shapes
: *
dtype0
Ф
batch_normalization_1_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name batch_normalization_1_31/gamma
Н
2batch_normalization_1_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1_31/gamma*
_output_shapes
: *
dtype0
Т
batch_normalization_1_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1_31/beta
Л
1batch_normalization_1_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1_31/beta*
_output_shapes
: *
dtype0
†
$batch_normalization_1_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$batch_normalization_1_31/moving_mean
Щ
8batch_normalization_1_31/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_1_31/moving_mean*
_output_shapes
: *
dtype0
®
(batch_normalization_1_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(batch_normalization_1_31/moving_variance
°
<batch_normalization_1_31/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_1_31/moving_variance*
_output_shapes
: *
dtype0
И
conv2d_2_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameconv2d_2_31/kernel
Б
&conv2d_2_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_2_31/kernel*&
_output_shapes
: @*
dtype0
x
conv2d_2_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_2_31/bias
q
$conv2d_2_31/bias/Read/ReadVariableOpReadVariableOpconv2d_2_31/bias*
_output_shapes
:@*
dtype0
Ф
batch_normalization_2_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name batch_normalization_2_31/gamma
Н
2batch_normalization_2_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2_31/gamma*
_output_shapes
:@*
dtype0
Т
batch_normalization_2_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_2_31/beta
Л
1batch_normalization_2_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2_31/beta*
_output_shapes
:@*
dtype0
†
$batch_normalization_2_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$batch_normalization_2_31/moving_mean
Щ
8batch_normalization_2_31/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_2_31/moving_mean*
_output_shapes
:@*
dtype0
®
(batch_normalization_2_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(batch_normalization_2_31/moving_variance
°
<batch_normalization_2_31/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_2_31/moving_variance*
_output_shapes
:@*
dtype0
И
conv2d_3_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv2d_3_31/kernel
Б
&conv2d_3_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_3_31/kernel*&
_output_shapes
:@@*
dtype0
x
conv2d_3_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_3_31/bias
q
$conv2d_3_31/bias/Read/ReadVariableOpReadVariableOpconv2d_3_31/bias*
_output_shapes
:@*
dtype0
Ф
batch_normalization_3_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name batch_normalization_3_31/gamma
Н
2batch_normalization_3_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3_31/gamma*
_output_shapes
:@*
dtype0
Т
batch_normalization_3_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_3_31/beta
Л
1batch_normalization_3_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3_31/beta*
_output_shapes
:@*
dtype0
†
$batch_normalization_3_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$batch_normalization_3_31/moving_mean
Щ
8batch_normalization_3_31/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_3_31/moving_mean*
_output_shapes
:@*
dtype0
®
(batch_normalization_3_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(batch_normalization_3_31/moving_variance
°
<batch_normalization_3_31/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_3_31/moving_variance*
_output_shapes
:@*
dtype0
Й
conv2d_4_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*#
shared_nameconv2d_4_31/kernel
В
&conv2d_4_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_4_31/kernel*'
_output_shapes
:@А*
dtype0
y
conv2d_4_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameconv2d_4_31/bias
r
$conv2d_4_31/bias/Read/ReadVariableOpReadVariableOpconv2d_4_31/bias*
_output_shapes	
:А*
dtype0
Х
batch_normalization_4_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name batch_normalization_4_31/gamma
О
2batch_normalization_4_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4_31/gamma*
_output_shapes	
:А*
dtype0
У
batch_normalization_4_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namebatch_normalization_4_31/beta
М
1batch_normalization_4_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4_31/beta*
_output_shapes	
:А*
dtype0
°
$batch_normalization_4_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$batch_normalization_4_31/moving_mean
Ъ
8batch_normalization_4_31/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_4_31/moving_mean*
_output_shapes	
:А*
dtype0
©
(batch_normalization_4_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(batch_normalization_4_31/moving_variance
Ґ
<batch_normalization_4_31/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_4_31/moving_variance*
_output_shapes	
:А*
dtype0
К
conv2d_5_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*#
shared_nameconv2d_5_31/kernel
Г
&conv2d_5_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_5_31/kernel*(
_output_shapes
:АА*
dtype0
y
conv2d_5_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*!
shared_nameconv2d_5_31/bias
r
$conv2d_5_31/bias/Read/ReadVariableOpReadVariableOpconv2d_5_31/bias*
_output_shapes	
:А*
dtype0
Х
batch_normalization_5_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name batch_normalization_5_31/gamma
О
2batch_normalization_5_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5_31/gamma*
_output_shapes	
:А*
dtype0
У
batch_normalization_5_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namebatch_normalization_5_31/beta
М
1batch_normalization_5_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5_31/beta*
_output_shapes	
:А*
dtype0
°
$batch_normalization_5_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$batch_normalization_5_31/moving_mean
Ъ
8batch_normalization_5_31/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_5_31/moving_mean*
_output_shapes	
:А*
dtype0
©
(batch_normalization_5_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(batch_normalization_5_31/moving_variance
Ґ
<batch_normalization_5_31/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_5_31/moving_variance*
_output_shapes	
:А*
dtype0
|
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА* 
shared_namedense_32/kernel
u
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel* 
_output_shapes
:
АА*
dtype0
s
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_32/bias
l
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes	
:А*
dtype0
Х
batch_normalization_6_31/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name batch_normalization_6_31/gamma
О
2batch_normalization_6_31/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6_31/gamma*
_output_shapes	
:А*
dtype0
У
batch_normalization_6_31/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namebatch_normalization_6_31/beta
М
1batch_normalization_6_31/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6_31/beta*
_output_shapes	
:А*
dtype0
°
$batch_normalization_6_31/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$batch_normalization_6_31/moving_mean
Ъ
8batch_normalization_6_31/moving_mean/Read/ReadVariableOpReadVariableOp$batch_normalization_6_31/moving_mean*
_output_shapes	
:А*
dtype0
©
(batch_normalization_6_31/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(batch_normalization_6_31/moving_variance
Ґ
<batch_normalization_6_31/moving_variance/Read/ReadVariableOpReadVariableOp(batch_normalization_6_31/moving_variance*
_output_shapes	
:А*
dtype0

dense_1_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А
*"
shared_namedense_1_31/kernel
x
%dense_1_31/kernel/Read/ReadVariableOpReadVariableOpdense_1_31/kernel*
_output_shapes
:	А
*
dtype0
v
dense_1_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_1_31/bias
o
#dense_1_31/bias/Read/ReadVariableOpReadVariableOpdense_1_31/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
–q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Лq
valueБqBюp Bчp
і
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
Ч
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
Ч
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
R
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
Ч
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Ч
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^	variables
_trainable_variables
`regularization_losses
a	keras_api
R
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
Ч
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
h

ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
Я
axis

Аgamma
	Бbeta
Вmoving_mean
Гmoving_variance
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
V
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
V
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
V
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
n
Фkernel
	Хbias
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
†
	Ъaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
V
£	variables
§trainable_variables
•regularization_losses
¶	keras_api
n
Іkernel
	®bias
©	variables
™trainable_variables
Ђregularization_losses
ђ	keras_api
 
в
0
1
%2
&3
'4
(5
-6
.7
48
59
610
711
D12
E13
K14
L15
M16
N17
S18
T19
Z20
[21
\22
]23
j24
k25
q26
r27
s28
t29
y30
z31
А32
Б33
В34
Г35
Ф36
Х37
Ы38
Ь39
Э40
Ю41
І42
®43
о
0
1
%2
&3
-4
.5
46
57
D8
E9
K10
L11
S12
T13
Z14
[15
j16
k17
q18
r19
y20
z21
А22
Б23
Ф24
Х25
Ы26
Ь27
І28
®29
 
≤
	variables
trainable_variables
regularization_losses
≠non_trainable_variables
Ѓlayers
 ѓlayer_regularization_losses
∞metrics
±layer_metrics
 
\Z
VARIABLE_VALUEconv2d_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≤
 	variables
!trainable_variables
"regularization_losses
≤layers
≥non_trainable_variables
 іlayer_regularization_losses
µmetrics
ґlayer_metrics
 
ge
VARIABLE_VALUEbatch_normalization_37/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_37/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_37/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_37/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
'2
(3

%0
&1
 
≤
)	variables
*trainable_variables
+regularization_losses
Јlayers
Єnon_trainable_variables
 єlayer_regularization_losses
Їmetrics
їlayer_metrics
^\
VARIABLE_VALUEconv2d_1_31/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_1_31/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
≤
/	variables
0trainable_variables
1regularization_losses
Љlayers
љnon_trainable_variables
 Њlayer_regularization_losses
њmetrics
јlayer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_1_31/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_1_31/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_1_31/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_1_31/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

40
51
62
73

40
51
 
≤
8	variables
9trainable_variables
:regularization_losses
Ѕlayers
¬non_trainable_variables
 √layer_regularization_losses
ƒmetrics
≈layer_metrics
 
 
 
≤
<	variables
=trainable_variables
>regularization_losses
∆layers
«non_trainable_variables
 »layer_regularization_losses
…metrics
 layer_metrics
 
 
 
≤
@	variables
Atrainable_variables
Bregularization_losses
Ћlayers
ћnon_trainable_variables
 Ќlayer_regularization_losses
ќmetrics
ѕlayer_metrics
^\
VARIABLE_VALUEconv2d_2_31/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_2_31/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
≤
F	variables
Gtrainable_variables
Hregularization_losses
–layers
—non_trainable_variables
 “layer_regularization_losses
”metrics
‘layer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_2_31/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_2_31/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_2_31/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_2_31/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
M2
N3

K0
L1
 
≤
O	variables
Ptrainable_variables
Qregularization_losses
’layers
÷non_trainable_variables
 „layer_regularization_losses
Ўmetrics
ўlayer_metrics
^\
VARIABLE_VALUEconv2d_3_31/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_3_31/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
≤
U	variables
Vtrainable_variables
Wregularization_losses
Џlayers
џnon_trainable_variables
 №layer_regularization_losses
Ёmetrics
ёlayer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_3_31/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_3_31/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_3_31/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_3_31/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Z0
[1
\2
]3

Z0
[1
 
≤
^	variables
_trainable_variables
`regularization_losses
яlayers
аnon_trainable_variables
 бlayer_regularization_losses
вmetrics
гlayer_metrics
 
 
 
≤
b	variables
ctrainable_variables
dregularization_losses
дlayers
еnon_trainable_variables
 жlayer_regularization_losses
зmetrics
иlayer_metrics
 
 
 
≤
f	variables
gtrainable_variables
hregularization_losses
йlayers
кnon_trainable_variables
 лlayer_regularization_losses
мmetrics
нlayer_metrics
^\
VARIABLE_VALUEconv2d_4_31/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2d_4_31/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
≤
l	variables
mtrainable_variables
nregularization_losses
оlayers
пnon_trainable_variables
 рlayer_regularization_losses
сmetrics
тlayer_metrics
 
ig
VARIABLE_VALUEbatch_normalization_4_31/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_normalization_4_31/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE$batch_normalization_4_31/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE(batch_normalization_4_31/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

q0
r1
s2
t3

q0
r1
 
≤
u	variables
vtrainable_variables
wregularization_losses
уlayers
фnon_trainable_variables
 хlayer_regularization_losses
цmetrics
чlayer_metrics
_]
VARIABLE_VALUEconv2d_5_31/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEconv2d_5_31/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

y0
z1

y0
z1
 
≤
{	variables
|trainable_variables
}regularization_losses
шlayers
щnon_trainable_variables
 ъlayer_regularization_losses
ыmetrics
ьlayer_metrics
 
jh
VARIABLE_VALUEbatch_normalization_5_31/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_5_31/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_5_31/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_5_31/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
А0
Б1
В2
Г3

А0
Б1
 
µ
Д	variables
Еtrainable_variables
Жregularization_losses
эlayers
юnon_trainable_variables
 €layer_regularization_losses
Аmetrics
Бlayer_metrics
 
 
 
µ
И	variables
Йtrainable_variables
Кregularization_losses
Вlayers
Гnon_trainable_variables
 Дlayer_regularization_losses
Еmetrics
Жlayer_metrics
 
 
 
µ
М	variables
Нtrainable_variables
Оregularization_losses
Зlayers
Иnon_trainable_variables
 Йlayer_regularization_losses
Кmetrics
Лlayer_metrics
 
 
 
µ
Р	variables
Сtrainable_variables
Тregularization_losses
Мlayers
Нnon_trainable_variables
 Оlayer_regularization_losses
Пmetrics
Рlayer_metrics
\Z
VARIABLE_VALUEdense_32/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_32/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Ф0
Х1

Ф0
Х1
 
µ
Ц	variables
Чtrainable_variables
Шregularization_losses
Сlayers
Тnon_trainable_variables
 Уlayer_regularization_losses
Фmetrics
Хlayer_metrics
 
jh
VARIABLE_VALUEbatch_normalization_6_31/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatch_normalization_6_31/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE$batch_normalization_6_31/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE(batch_normalization_6_31/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ы0
Ь1
Э2
Ю3

Ы0
Ь1
 
µ
Я	variables
†trainable_variables
°regularization_losses
Цlayers
Чnon_trainable_variables
 Шlayer_regularization_losses
Щmetrics
Ъlayer_metrics
 
 
 
µ
£	variables
§trainable_variables
•regularization_losses
Ыlayers
Ьnon_trainable_variables
 Эlayer_regularization_losses
Юmetrics
Яlayer_metrics
^\
VARIABLE_VALUEdense_1_31/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_1_31/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

І0
®1

І0
®1
 
µ
©	variables
™trainable_variables
Ђregularization_losses
†layers
°non_trainable_variables
 Ґlayer_regularization_losses
£metrics
§layer_metrics
j
'0
(1
62
73
M4
N5
\6
]7
s8
t9
В10
Г11
Э12
Ю13
Ѓ
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
11
12
13
14
15
16
17
18
19
20
21
22
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
'0
(1
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
60
71
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

M0
N1
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
\0
]1
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

s0
t1
 
 
 
 
 
 
 
 
 

В0
Г1
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

Э0
Ю1
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
Л
serving_default_input_11Placeholder*/
_output_shapes
:€€€€€€€€€  *
dtype0*$
shape:€€€€€€€€€  
п
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11conv2d_36/kernelconv2d_36/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_varianceconv2d_1_31/kernelconv2d_1_31/biasbatch_normalization_1_31/gammabatch_normalization_1_31/beta$batch_normalization_1_31/moving_mean(batch_normalization_1_31/moving_varianceconv2d_2_31/kernelconv2d_2_31/biasbatch_normalization_2_31/gammabatch_normalization_2_31/beta$batch_normalization_2_31/moving_mean(batch_normalization_2_31/moving_varianceconv2d_3_31/kernelconv2d_3_31/biasbatch_normalization_3_31/gammabatch_normalization_3_31/beta$batch_normalization_3_31/moving_mean(batch_normalization_3_31/moving_varianceconv2d_4_31/kernelconv2d_4_31/biasbatch_normalization_4_31/gammabatch_normalization_4_31/beta$batch_normalization_4_31/moving_mean(batch_normalization_4_31/moving_varianceconv2d_5_31/kernelconv2d_5_31/biasbatch_normalization_5_31/gammabatch_normalization_5_31/beta$batch_normalization_5_31/moving_mean(batch_normalization_5_31/moving_variancedense_32/kerneldense_32/bias(batch_normalization_6_31/moving_variancebatch_normalization_6_31/gamma$batch_normalization_6_31/moving_meanbatch_normalization_6_31/betadense_1_31/kerneldense_1_31/bias*8
Tin1
/2-*
Tout
2*'
_output_shapes
:€€€€€€€€€
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,**
config_proto

CPU

GPU 2J 8*.
f)R'
%__inference_signature_wrapper_1640343
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ы
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp0batch_normalization_37/gamma/Read/ReadVariableOp/batch_normalization_37/beta/Read/ReadVariableOp6batch_normalization_37/moving_mean/Read/ReadVariableOp:batch_normalization_37/moving_variance/Read/ReadVariableOp&conv2d_1_31/kernel/Read/ReadVariableOp$conv2d_1_31/bias/Read/ReadVariableOp2batch_normalization_1_31/gamma/Read/ReadVariableOp1batch_normalization_1_31/beta/Read/ReadVariableOp8batch_normalization_1_31/moving_mean/Read/ReadVariableOp<batch_normalization_1_31/moving_variance/Read/ReadVariableOp&conv2d_2_31/kernel/Read/ReadVariableOp$conv2d_2_31/bias/Read/ReadVariableOp2batch_normalization_2_31/gamma/Read/ReadVariableOp1batch_normalization_2_31/beta/Read/ReadVariableOp8batch_normalization_2_31/moving_mean/Read/ReadVariableOp<batch_normalization_2_31/moving_variance/Read/ReadVariableOp&conv2d_3_31/kernel/Read/ReadVariableOp$conv2d_3_31/bias/Read/ReadVariableOp2batch_normalization_3_31/gamma/Read/ReadVariableOp1batch_normalization_3_31/beta/Read/ReadVariableOp8batch_normalization_3_31/moving_mean/Read/ReadVariableOp<batch_normalization_3_31/moving_variance/Read/ReadVariableOp&conv2d_4_31/kernel/Read/ReadVariableOp$conv2d_4_31/bias/Read/ReadVariableOp2batch_normalization_4_31/gamma/Read/ReadVariableOp1batch_normalization_4_31/beta/Read/ReadVariableOp8batch_normalization_4_31/moving_mean/Read/ReadVariableOp<batch_normalization_4_31/moving_variance/Read/ReadVariableOp&conv2d_5_31/kernel/Read/ReadVariableOp$conv2d_5_31/bias/Read/ReadVariableOp2batch_normalization_5_31/gamma/Read/ReadVariableOp1batch_normalization_5_31/beta/Read/ReadVariableOp8batch_normalization_5_31/moving_mean/Read/ReadVariableOp<batch_normalization_5_31/moving_variance/Read/ReadVariableOp#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp2batch_normalization_6_31/gamma/Read/ReadVariableOp1batch_normalization_6_31/beta/Read/ReadVariableOp8batch_normalization_6_31/moving_mean/Read/ReadVariableOp<batch_normalization_6_31/moving_variance/Read/ReadVariableOp%dense_1_31/kernel/Read/ReadVariableOp#dense_1_31/bias/Read/ReadVariableOpConst*9
Tin2
02.*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference__traced_save_1642397
¶
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_36/kernelconv2d_36/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_varianceconv2d_1_31/kernelconv2d_1_31/biasbatch_normalization_1_31/gammabatch_normalization_1_31/beta$batch_normalization_1_31/moving_mean(batch_normalization_1_31/moving_varianceconv2d_2_31/kernelconv2d_2_31/biasbatch_normalization_2_31/gammabatch_normalization_2_31/beta$batch_normalization_2_31/moving_mean(batch_normalization_2_31/moving_varianceconv2d_3_31/kernelconv2d_3_31/biasbatch_normalization_3_31/gammabatch_normalization_3_31/beta$batch_normalization_3_31/moving_mean(batch_normalization_3_31/moving_varianceconv2d_4_31/kernelconv2d_4_31/biasbatch_normalization_4_31/gammabatch_normalization_4_31/beta$batch_normalization_4_31/moving_mean(batch_normalization_4_31/moving_varianceconv2d_5_31/kernelconv2d_5_31/biasbatch_normalization_5_31/gammabatch_normalization_5_31/beta$batch_normalization_5_31/moving_mean(batch_normalization_5_31/moving_variancedense_32/kerneldense_32/biasbatch_normalization_6_31/gammabatch_normalization_6_31/beta$batch_normalization_6_31/moving_mean(batch_normalization_6_31/moving_variancedense_1_31/kerneldense_1_31/bias*8
Tin1
/2-*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__traced_restore_1642541зу
р
ђ
D__inference_dense_1_layer_call_and_return_conditional_losses_1642229

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Р
E
)__inference_dropout_layer_call_fn_1641340

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_16390432
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
…
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641476

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@:::::W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
†$
ў
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641620

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Т
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1638064

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ю
Л
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1638680

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
К
Х
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1638832

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А:::::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1638900

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   :::::W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ў
У
%__inference_signature_wrapper_1640343
input_11
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identityИҐStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*'
_output_shapes
:€€€€€€€€€
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,**
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__wrapped_model_16377792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_11:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
≥

≠
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1637939

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
E
)__inference_flatten_layer_call_fn_1642053

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16394802
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≤$
ў
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641734

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Љ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpњ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ц
Ъ
,__inference_sequential_layer_call_fn_1639932
input_11
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*'
_output_shapes
:€€€€€€€€€
*@
_read_only_resource_inputs"
 	
 !"%&)*+,**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16398412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_11:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
†$
ў
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641458

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓf
№
 __inference__traced_save_1642397
file_prefix/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop;
7savev2_batch_normalization_37_gamma_read_readvariableop:
6savev2_batch_normalization_37_beta_read_readvariableopA
=savev2_batch_normalization_37_moving_mean_read_readvariableopE
Asavev2_batch_normalization_37_moving_variance_read_readvariableop1
-savev2_conv2d_1_31_kernel_read_readvariableop/
+savev2_conv2d_1_31_bias_read_readvariableop=
9savev2_batch_normalization_1_31_gamma_read_readvariableop<
8savev2_batch_normalization_1_31_beta_read_readvariableopC
?savev2_batch_normalization_1_31_moving_mean_read_readvariableopG
Csavev2_batch_normalization_1_31_moving_variance_read_readvariableop1
-savev2_conv2d_2_31_kernel_read_readvariableop/
+savev2_conv2d_2_31_bias_read_readvariableop=
9savev2_batch_normalization_2_31_gamma_read_readvariableop<
8savev2_batch_normalization_2_31_beta_read_readvariableopC
?savev2_batch_normalization_2_31_moving_mean_read_readvariableopG
Csavev2_batch_normalization_2_31_moving_variance_read_readvariableop1
-savev2_conv2d_3_31_kernel_read_readvariableop/
+savev2_conv2d_3_31_bias_read_readvariableop=
9savev2_batch_normalization_3_31_gamma_read_readvariableop<
8savev2_batch_normalization_3_31_beta_read_readvariableopC
?savev2_batch_normalization_3_31_moving_mean_read_readvariableopG
Csavev2_batch_normalization_3_31_moving_variance_read_readvariableop1
-savev2_conv2d_4_31_kernel_read_readvariableop/
+savev2_conv2d_4_31_bias_read_readvariableop=
9savev2_batch_normalization_4_31_gamma_read_readvariableop<
8savev2_batch_normalization_4_31_beta_read_readvariableopC
?savev2_batch_normalization_4_31_moving_mean_read_readvariableopG
Csavev2_batch_normalization_4_31_moving_variance_read_readvariableop1
-savev2_conv2d_5_31_kernel_read_readvariableop/
+savev2_conv2d_5_31_bias_read_readvariableop=
9savev2_batch_normalization_5_31_gamma_read_readvariableop<
8savev2_batch_normalization_5_31_beta_read_readvariableopC
?savev2_batch_normalization_5_31_moving_mean_read_readvariableopG
Csavev2_batch_normalization_5_31_moving_variance_read_readvariableop.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop=
9savev2_batch_normalization_6_31_gamma_read_readvariableop<
8savev2_batch_normalization_6_31_beta_read_readvariableopC
?savev2_batch_normalization_6_31_moving_mean_read_readvariableopG
Csavev2_batch_normalization_6_31_moving_variance_read_readvariableop0
,savev2_dense_1_31_kernel_read_readvariableop.
*savev2_dense_1_31_bias_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4eacf04ea0c14a7f97ea0f9a75b436a8/part2	
Const_1Л
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameо
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*А
valueцBу,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesа
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesА
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop7savev2_batch_normalization_37_gamma_read_readvariableop6savev2_batch_normalization_37_beta_read_readvariableop=savev2_batch_normalization_37_moving_mean_read_readvariableopAsavev2_batch_normalization_37_moving_variance_read_readvariableop-savev2_conv2d_1_31_kernel_read_readvariableop+savev2_conv2d_1_31_bias_read_readvariableop9savev2_batch_normalization_1_31_gamma_read_readvariableop8savev2_batch_normalization_1_31_beta_read_readvariableop?savev2_batch_normalization_1_31_moving_mean_read_readvariableopCsavev2_batch_normalization_1_31_moving_variance_read_readvariableop-savev2_conv2d_2_31_kernel_read_readvariableop+savev2_conv2d_2_31_bias_read_readvariableop9savev2_batch_normalization_2_31_gamma_read_readvariableop8savev2_batch_normalization_2_31_beta_read_readvariableop?savev2_batch_normalization_2_31_moving_mean_read_readvariableopCsavev2_batch_normalization_2_31_moving_variance_read_readvariableop-savev2_conv2d_3_31_kernel_read_readvariableop+savev2_conv2d_3_31_bias_read_readvariableop9savev2_batch_normalization_3_31_gamma_read_readvariableop8savev2_batch_normalization_3_31_beta_read_readvariableop?savev2_batch_normalization_3_31_moving_mean_read_readvariableopCsavev2_batch_normalization_3_31_moving_variance_read_readvariableop-savev2_conv2d_4_31_kernel_read_readvariableop+savev2_conv2d_4_31_bias_read_readvariableop9savev2_batch_normalization_4_31_gamma_read_readvariableop8savev2_batch_normalization_4_31_beta_read_readvariableop?savev2_batch_normalization_4_31_moving_mean_read_readvariableopCsavev2_batch_normalization_4_31_moving_variance_read_readvariableop-savev2_conv2d_5_31_kernel_read_readvariableop+savev2_conv2d_5_31_bias_read_readvariableop9savev2_batch_normalization_5_31_gamma_read_readvariableop8savev2_batch_normalization_5_31_beta_read_readvariableop?savev2_batch_normalization_5_31_moving_mean_read_readvariableopCsavev2_batch_normalization_5_31_moving_variance_read_readvariableop*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop9savev2_batch_normalization_6_31_gamma_read_readvariableop8savev2_batch_normalization_6_31_beta_read_readvariableop?savev2_batch_normalization_6_31_moving_mean_read_readvariableopCsavev2_batch_normalization_6_31_moving_variance_read_readvariableop,savev2_dense_1_31_kernel_read_readvariableop*savev2_dense_1_31_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ж
_input_shapesф
с: : : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:@А:А:А:А:А:А:АА:А:А:А:А:А:
АА:А:А:А:А:А:	А
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:! 

_output_shapes	
:А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:!#

_output_shapes	
:А:!$

_output_shapes	
:А:&%"
 
_output_shapes
:
АА:!&

_output_shapes	
:А:!'

_output_shapes	
:А:!(

_output_shapes	
:А:!)

_output_shapes	
:А:!*

_output_shapes	
:А:%+!

_output_shapes
:	А
: ,

_output_shapes
:
:-

_output_shapes
: 
ч
™
7__inference_batch_normalization_2_layer_call_fn_1641427

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16382242
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ
™
7__inference_batch_normalization_5_layer_call_fn_1641927

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16386492
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ш
G
+__inference_dropout_3_layer_call_fn_1642218

inputs
identity£
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_16395672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
р
Ш
,__inference_sequential_layer_call_fn_1640896

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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identityИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*'
_output_shapes
:€€€€€€€€€
*@
_read_only_resource_inputs"
 	
 !"%&)*+,**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16398412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
є

≠
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1638407

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
ReluБ
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
’
Л
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1639318

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ќ
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1639567

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
и$
ў
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641545

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
†
d
+__inference_dropout_1_layer_call_fn_1641686

inputs
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16392472
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
и$
ў
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641383

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
’
Л
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641752

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
у
®
5__inference_batch_normalization_layer_call_fn_1641076

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16379162
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
…
Л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641638

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@:::::W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
†$
ў
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1638971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
й
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1641681

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
≤$
ў
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1639389

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Љ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpњ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Д
d
+__inference_dropout_3_layer_call_fn_1642213

inputs
identityИҐStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_16395622
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
©
®
5__inference_batch_normalization_layer_call_fn_1641138

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16388822
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≤$
ў
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1639300

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Љ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpњ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ$
ў
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1638649

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp—
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Т
Л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1638372

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Р
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641050

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з
b
D__inference_dropout_layer_call_and_return_conditional_losses_1639043

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ю
Л
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641827

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ$
ў
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1638501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp—
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≥
™
7__inference_batch_normalization_4_layer_call_fn_1641778

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16393182
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
…
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1638989

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   :::::W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
…
Л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1639198

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@:::::W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ы
™
7__inference_batch_normalization_5_layer_call_fn_1641940

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16386802
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≥

≠
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1638099

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
х
™
7__inference_batch_normalization_1_layer_call_fn_1641225

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16380332
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч
™
7__inference_batch_normalization_3_layer_call_fn_1641589

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16383722
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
А
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1638081

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ці
“
G__inference_sequential_layer_call_and_return_conditional_losses_1640803

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource?
;batch_normalization_6_batchnorm_mul_readvariableop_resource=
9batch_normalization_6_batchnorm_readvariableop_1_resource=
9batch_normalization_6_batchnorm_readvariableop_2_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИ™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpЄ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d/Relu∞
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpй
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1’
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 2&
$batch_normalization/FusedBatchNormV3∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpа
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_1/Reluґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1г
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3“
max_pooling2d/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolК
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/Identity∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp—
conv2d_2/Conv2DConv2Ddropout/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/Reluґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1г
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpв
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_3/Reluґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1г
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3÷
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolР
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout_1/Identity±
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_4/Conv2D/ReadVariableOp‘
conv2d_4/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_4/Conv2D®
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp≠
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_4/ReluЈ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_4/ReadVariableOpљ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_4/ReadVariableOp_1к
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpр
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1и
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3≤
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_5/Conv2D/ReadVariableOpг
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_5/Conv2D®
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp≠
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_5/BiasAdd|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_5/ReluЈ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_5/ReadVariableOpљ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_5/ReadVariableOp_1к
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpр
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1и
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3„
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolС
dropout_2/IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_2/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstХ
flatten/ReshapeReshapedropout_2/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

dense/Relu’
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yб
#batch_normalization_6/batchnorm/addAddV26batch_normalization_6/batchnorm/ReadVariableOp:value:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/add¶
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/Rsqrtб
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpё
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/mulЋ
%batch_normalization_6/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_6/batchnorm/mul_1џ
0batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_1ё
%batch_normalization_6/batchnorm/mul_2Mul8batch_normalization_6/batchnorm/ReadVariableOp_1:value:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/mul_2џ
0batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype022
0batch_normalization_6/batchnorm/ReadVariableOp_2№
#batch_normalization_6/batchnorm/subSub8batch_normalization_6/batchnorm/ReadVariableOp_2:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/subё
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_6/batchnorm/add_1Т
dropout_3/IdentityIdentity)batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_3/Identity¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_1/MatMul/ReadVariableOp†
dense_1/MatMulMatMuldropout_3/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_1/Softmaxm
IdentityIdentitydense_1/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  :::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
а

*__inference_conv2d_1_layer_call_fn_1637949

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_16379392
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Њ
`
D__inference_flatten_layer_call_and_return_conditional_losses_1639480

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
щ
™
7__inference_batch_normalization_4_layer_call_fn_1641840

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16385012
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
™
7__inference_batch_normalization_1_layer_call_fn_1641313

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16389892
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ƒ
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1639247

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЉ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y∆
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€@2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ю$
„
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641107

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641194

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ь
b
)__inference_dropout_layer_call_fn_1641335

inputs
identityИҐStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_16390382
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ы
™
7__inference_batch_normalization_4_layer_call_fn_1641853

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16385322
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
…
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641287

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   :::::W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≠
™
7__inference_batch_normalization_1_layer_call_fn_1641300

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16389712
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
а

*__inference_conv2d_3_layer_call_fn_1638257

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_16382472
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ƒ
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_1641676

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЉ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2
dropout/GreaterEqual/y∆
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€@2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
х
™
7__inference_batch_normalization_2_layer_call_fn_1641414

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16381932
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ч
™
7__inference_batch_normalization_1_layer_call_fn_1641238

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16380642
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ю
Ш
,__inference_sequential_layer_call_fn_1640989

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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*'
_output_shapes
:€€€€€€€€€
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16400492
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
…
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1639109

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@:::::W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1638193

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ$
ў
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641896

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp—
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Н
M
1__inference_max_pooling2d_2_layer_call_fn_1638703

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_16386972
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Л
e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1639562

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≥
™
7__inference_batch_normalization_5_layer_call_fn_1642015

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16394072
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¬
c
D__inference_dropout_layer_call_and_return_conditional_losses_1639038

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЉ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y∆
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€ 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
∆А
ц
G__inference_sequential_layer_call_and_return_conditional_losses_1639608
input_11
conv2d_1638847
conv2d_1638849
batch_normalization_1638927
batch_normalization_1638929
batch_normalization_1638931
batch_normalization_1638933
conv2d_1_1638936
conv2d_1_1638938!
batch_normalization_1_1639016!
batch_normalization_1_1639018!
batch_normalization_1_1639020!
batch_normalization_1_1639022
conv2d_2_1639056
conv2d_2_1639058!
batch_normalization_2_1639136!
batch_normalization_2_1639138!
batch_normalization_2_1639140!
batch_normalization_2_1639142
conv2d_3_1639145
conv2d_3_1639147!
batch_normalization_3_1639225!
batch_normalization_3_1639227!
batch_normalization_3_1639229!
batch_normalization_3_1639231
conv2d_4_1639265
conv2d_4_1639267!
batch_normalization_4_1639345!
batch_normalization_4_1639347!
batch_normalization_4_1639349!
batch_normalization_4_1639351
conv2d_5_1639354
conv2d_5_1639356!
batch_normalization_5_1639434!
batch_normalization_5_1639436!
batch_normalization_5_1639438!
batch_normalization_5_1639440
dense_1639510
dense_1639512!
batch_normalization_6_1639541!
batch_normalization_6_1639543!
batch_normalization_6_1639545!
batch_normalization_6_1639547
dense_1_1639602
dense_1_1639604
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallх
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_11conv2d_1638847conv2d_1638849*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_16377912 
conv2d/StatefulPartitionedCallС
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1638927batch_normalization_1638929batch_normalization_1638931batch_normalization_1638933*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16388822-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_1638936conv2d_1_1638938*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_16379392"
 conv2d_1/StatefulPartitionedCall°
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1639016batch_normalization_1_1639018batch_normalization_1_1639020batch_normalization_1_1639022*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16389712/
-batch_normalization_1/StatefulPartitionedCallъ
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_16380812
max_pooling2d/PartitionedCallр
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_16390382!
dropout/StatefulPartitionedCallЯ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_2_1639056conv2d_2_1639058*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_16380992"
 conv2d_2/StatefulPartitionedCall°
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1639136batch_normalization_2_1639138batch_normalization_2_1639140batch_normalization_2_1639142*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16390912/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_1639145conv2d_3_1639147*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_16382472"
 conv2d_3/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1639225batch_normalization_3_1639227batch_normalization_3_1639229batch_normalization_3_1639231*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16391802/
-batch_normalization_3/StatefulPartitionedCallА
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_16383892!
max_pooling2d_1/PartitionedCallЪ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16392472#
!dropout_1/StatefulPartitionedCallҐ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_4_1639265conv2d_4_1639267*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16384072"
 conv2d_4/StatefulPartitionedCallҐ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_1639345batch_normalization_4_1639347batch_normalization_4_1639349batch_normalization_4_1639351*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16393002/
-batch_normalization_4/StatefulPartitionedCallЃ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_5_1639354conv2d_5_1639356*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16385552"
 conv2d_5/StatefulPartitionedCallҐ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_1639434batch_normalization_5_1639436batch_normalization_5_1639438batch_normalization_5_1639440*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16393892/
-batch_normalization_5/StatefulPartitionedCallБ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_16386972!
max_pooling2d_2/PartitionedCallЭ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_16394562#
!dropout_2/StatefulPartitionedCall’
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16394802
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1639510dense_1639512*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16394992
dense/StatefulPartitionedCallЧ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_6_1639541batch_normalization_6_1639543batch_normalization_6_1639545batch_normalization_6_1639547*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_16387992/
-batch_normalization_6/StatefulPartitionedCall£
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_16395622#
!dropout_3/StatefulPartitionedCallФ
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_1_1639602dense_1_1639604*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16395912!
dense_1/StatefulPartitionedCallк
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_11:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
В
h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1638697

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
™
B__inference_dense_layer_call_and_return_conditional_losses_1639499

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
¬
c
D__inference_dropout_layer_call_and_return_conditional_losses_1641325

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЉ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2
dropout/GreaterEqual/y∆
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€ 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Т
Л
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641212

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
х
™
7__inference_batch_normalization_3_layer_call_fn_1641576

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16383412
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
’
Л
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1639407

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
±
™
7__inference_batch_normalization_5_layer_call_fn_1642002

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16393892
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
в

*__inference_conv2d_4_layer_call_fn_1638417

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16384072
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
§
d
+__inference_dropout_2_layer_call_fn_1642037

inputs
identityИҐStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_16394562
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
№
}
(__inference_conv2d_layer_call_fn_1637801

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_16377912
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
з
b
D__inference_dropout_layer_call_and_return_conditional_losses_1641330

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
јА
ф
G__inference_sequential_layer_call_and_return_conditional_losses_1639841

inputs
conv2d_1639729
conv2d_1639731
batch_normalization_1639734
batch_normalization_1639736
batch_normalization_1639738
batch_normalization_1639740
conv2d_1_1639743
conv2d_1_1639745!
batch_normalization_1_1639748!
batch_normalization_1_1639750!
batch_normalization_1_1639752!
batch_normalization_1_1639754
conv2d_2_1639759
conv2d_2_1639761!
batch_normalization_2_1639764!
batch_normalization_2_1639766!
batch_normalization_2_1639768!
batch_normalization_2_1639770
conv2d_3_1639773
conv2d_3_1639775!
batch_normalization_3_1639778!
batch_normalization_3_1639780!
batch_normalization_3_1639782!
batch_normalization_3_1639784
conv2d_4_1639789
conv2d_4_1639791!
batch_normalization_4_1639794!
batch_normalization_4_1639796!
batch_normalization_4_1639798!
batch_normalization_4_1639800
conv2d_5_1639803
conv2d_5_1639805!
batch_normalization_5_1639808!
batch_normalization_5_1639810!
batch_normalization_5_1639812!
batch_normalization_5_1639814
dense_1639820
dense_1639822!
batch_normalization_6_1639825!
batch_normalization_6_1639827!
batch_normalization_6_1639829!
batch_normalization_6_1639831
dense_1_1639835
dense_1_1639837
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdropout/StatefulPartitionedCallҐ!dropout_1/StatefulPartitionedCallҐ!dropout_2/StatefulPartitionedCallҐ!dropout_3/StatefulPartitionedCallу
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1639729conv2d_1639731*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_16377912 
conv2d/StatefulPartitionedCallС
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1639734batch_normalization_1639736batch_normalization_1639738batch_normalization_1639740*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16388822-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_1639743conv2d_1_1639745*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_16379392"
 conv2d_1/StatefulPartitionedCall°
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1639748batch_normalization_1_1639750batch_normalization_1_1639752batch_normalization_1_1639754*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16389712/
-batch_normalization_1/StatefulPartitionedCallъ
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_16380812
max_pooling2d/PartitionedCallр
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_16390382!
dropout/StatefulPartitionedCallЯ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_2_1639759conv2d_2_1639761*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_16380992"
 conv2d_2/StatefulPartitionedCall°
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1639764batch_normalization_2_1639766batch_normalization_2_1639768batch_normalization_2_1639770*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16390912/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_1639773conv2d_3_1639775*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_16382472"
 conv2d_3/StatefulPartitionedCall°
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1639778batch_normalization_3_1639780batch_normalization_3_1639782batch_normalization_3_1639784*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16391802/
-batch_normalization_3/StatefulPartitionedCallА
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_16383892!
max_pooling2d_1/PartitionedCallЪ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16392472#
!dropout_1/StatefulPartitionedCallҐ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_4_1639789conv2d_4_1639791*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16384072"
 conv2d_4/StatefulPartitionedCallҐ
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_1639794batch_normalization_4_1639796batch_normalization_4_1639798batch_normalization_4_1639800*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16393002/
-batch_normalization_4/StatefulPartitionedCallЃ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_5_1639803conv2d_5_1639805*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16385552"
 conv2d_5/StatefulPartitionedCallҐ
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_1639808batch_normalization_5_1639810batch_normalization_5_1639812batch_normalization_5_1639814*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16393892/
-batch_normalization_5/StatefulPartitionedCallБ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_16386972!
max_pooling2d_2/PartitionedCallЭ
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_16394562#
!dropout_2/StatefulPartitionedCall’
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16394802
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1639820dense_1639822*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16394992
dense/StatefulPartitionedCallЧ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_6_1639825batch_normalization_6_1639827batch_normalization_6_1639829batch_normalization_6_1639831*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_16387992/
-batch_normalization_6/StatefulPartitionedCall£
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_16395622#
!dropout_3/StatefulPartitionedCallФ
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_1_1639835dense_1_1639837*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16395912!
dense_1/StatefulPartitionedCallк
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
Ю
Л
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641914

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
±

Ђ
C__inference_conv2d_layer_call_and_return_conditional_losses_1637791

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
≤z
и
G__inference_sequential_layer_call_and_return_conditional_losses_1639723
input_11
conv2d_1639611
conv2d_1639613
batch_normalization_1639616
batch_normalization_1639618
batch_normalization_1639620
batch_normalization_1639622
conv2d_1_1639625
conv2d_1_1639627!
batch_normalization_1_1639630!
batch_normalization_1_1639632!
batch_normalization_1_1639634!
batch_normalization_1_1639636
conv2d_2_1639641
conv2d_2_1639643!
batch_normalization_2_1639646!
batch_normalization_2_1639648!
batch_normalization_2_1639650!
batch_normalization_2_1639652
conv2d_3_1639655
conv2d_3_1639657!
batch_normalization_3_1639660!
batch_normalization_3_1639662!
batch_normalization_3_1639664!
batch_normalization_3_1639666
conv2d_4_1639671
conv2d_4_1639673!
batch_normalization_4_1639676!
batch_normalization_4_1639678!
batch_normalization_4_1639680!
batch_normalization_4_1639682
conv2d_5_1639685
conv2d_5_1639687!
batch_normalization_5_1639690!
batch_normalization_5_1639692!
batch_normalization_5_1639694!
batch_normalization_5_1639696
dense_1639702
dense_1639704!
batch_normalization_6_1639707!
batch_normalization_6_1639709!
batch_normalization_6_1639711!
batch_normalization_6_1639713
dense_1_1639717
dense_1_1639719
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallх
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_11conv2d_1639611conv2d_1639613*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_16377912 
conv2d/StatefulPartitionedCallУ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1639616batch_normalization_1639618batch_normalization_1639620batch_normalization_1639622*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16389002-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_1639625conv2d_1_1639627*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_16379392"
 conv2d_1/StatefulPartitionedCall£
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1639630batch_normalization_1_1639632batch_normalization_1_1639634batch_normalization_1_1639636*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16389892/
-batch_normalization_1/StatefulPartitionedCallъ
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_16380812
max_pooling2d/PartitionedCallЎ
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_16390432
dropout/PartitionedCallЧ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_2_1639641conv2d_2_1639643*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_16380992"
 conv2d_2/StatefulPartitionedCall£
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1639646batch_normalization_2_1639648batch_normalization_2_1639650batch_normalization_2_1639652*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16391092/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_1639655conv2d_3_1639657*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_16382472"
 conv2d_3/StatefulPartitionedCall£
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1639660batch_normalization_3_1639662batch_normalization_3_1639664batch_normalization_3_1639666*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16391982/
-batch_normalization_3/StatefulPartitionedCallА
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_16383892!
max_pooling2d_1/PartitionedCallа
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16392522
dropout_1/PartitionedCallЪ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_4_1639671conv2d_4_1639673*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16384072"
 conv2d_4/StatefulPartitionedCall§
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_1639676batch_normalization_4_1639678batch_normalization_4_1639680batch_normalization_4_1639682*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16393182/
-batch_normalization_4/StatefulPartitionedCallЃ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_5_1639685conv2d_5_1639687*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16385552"
 conv2d_5/StatefulPartitionedCall§
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_1639690batch_normalization_5_1639692batch_normalization_5_1639694batch_normalization_5_1639696*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16394072/
-batch_normalization_5/StatefulPartitionedCallБ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_16386972!
max_pooling2d_2/PartitionedCallб
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_16394612
dropout_2/PartitionedCallЌ
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16394802
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1639702dense_1639704*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16394992
dense/StatefulPartitionedCallЩ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_6_1639707batch_normalization_6_1639709batch_normalization_6_1639711batch_normalization_6_1639713*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_16388322/
-batch_normalization_6/StatefulPartitionedCallз
dropout_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_16395672
dropout_3/PartitionedCallМ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_1639717dense_1_1639719*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16395912!
dense_1/StatefulPartitionedCall№
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_11:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
Т
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1638224

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
с
®
5__inference_batch_normalization_layer_call_fn_1641063

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16378852
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
м
™
B__inference_dense_layer_call_and_return_conditional_losses_1642064

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Л
e
F__inference_dropout_3_layer_call_and_return_conditional_losses_1642203

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yњ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
О”
У
"__inference__wrapped_model_1637779
input_114
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource<
8sequential_batch_normalization_1_readvariableop_resource>
:sequential_batch_normalization_1_readvariableop_1_resourceM
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource<
8sequential_batch_normalization_2_readvariableop_resource>
:sequential_batch_normalization_2_readvariableop_1_resourceM
Isequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource<
8sequential_batch_normalization_3_readvariableop_resource>
:sequential_batch_normalization_3_readvariableop_1_resourceM
Isequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_4_conv2d_readvariableop_resource7
3sequential_conv2d_4_biasadd_readvariableop_resource<
8sequential_batch_normalization_4_readvariableop_resource>
:sequential_batch_normalization_4_readvariableop_1_resourceM
Isequential_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource6
2sequential_conv2d_5_conv2d_readvariableop_resource7
3sequential_conv2d_5_biasadd_readvariableop_resource<
8sequential_batch_normalization_5_readvariableop_resource>
:sequential_batch_normalization_5_readvariableop_1_resourceM
Isequential_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceO
Ksequential_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resourceF
Bsequential_batch_normalization_6_batchnorm_readvariableop_resourceJ
Fsequential_batch_normalization_6_batchnorm_mul_readvariableop_resourceH
Dsequential_batch_normalization_6_batchnorm_readvariableop_1_resourceH
Dsequential_batch_normalization_6_batchnorm_readvariableop_2_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityИЋ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpџ
sequential/conv2d/Conv2DConv2Dinput_11/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2
sequential/conv2d/Conv2D¬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp–
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
sequential/conv2d/BiasAddЦ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
sequential/conv2d/Relu—
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential/batch_normalization/ReadVariableOp„
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype021
/sequential/batch_normalization/ReadVariableOp_1Д
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpК
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3$sequential/conv2d/Relu:activations:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3—
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOpМ
sequential/conv2d_1/Conv2DConv2D3sequential/batch_normalization/FusedBatchNormV3:y:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2
sequential/conv2d_1/Conv2D»
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpЎ
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
sequential/conv2d_1/BiasAddЬ
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
sequential/conv2d_1/Relu„
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential/batch_normalization_1/ReadVariableOpЁ
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype023
1sequential/batch_normalization_1/ReadVariableOp_1К
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02B
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02D
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1∞
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_1/Relu:activations:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_1/FusedBatchNormV3у
 sequential/max_pooling2d/MaxPoolMaxPool5sequential/batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPoolЂ
sequential/dropout/IdentityIdentity)sequential/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential/dropout/Identity—
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOpэ
sequential/conv2d_2/Conv2DConv2D$sequential/dropout/Identity:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
sequential/conv2d_2/Conv2D»
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpЎ
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_2/BiasAddЬ
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_2/Relu„
/sequential/batch_normalization_2/ReadVariableOpReadVariableOp8sequential_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/batch_normalization_2/ReadVariableOpЁ
1sequential/batch_normalization_2/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1sequential/batch_normalization_2/ReadVariableOp_1К
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1∞
1sequential/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_2/Relu:activations:07sequential/batch_normalization_2/ReadVariableOp:value:09sequential/batch_normalization_2/ReadVariableOp_1:value:0Hsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_2/FusedBatchNormV3—
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_3/Conv2D/ReadVariableOpО
sequential/conv2d_3/Conv2DConv2D5sequential/batch_normalization_2/FusedBatchNormV3:y:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
sequential/conv2d_3/Conv2D»
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_3/BiasAdd/ReadVariableOpЎ
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_3/BiasAddЬ
sequential/conv2d_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_3/Relu„
/sequential/batch_normalization_3/ReadVariableOpReadVariableOp8sequential_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential/batch_normalization_3/ReadVariableOpЁ
1sequential/batch_normalization_3/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype023
1sequential/batch_normalization_3/ReadVariableOp_1К
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02B
@sequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOpР
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02D
Bsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1∞
1sequential/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_3/Relu:activations:07sequential/batch_normalization_3/ReadVariableOp:value:09sequential/batch_normalization_3/ReadVariableOp_1:value:0Hsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_3/FusedBatchNormV3ч
"sequential/max_pooling2d_1/MaxPoolMaxPool5sequential/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_1/MaxPool±
sequential/dropout_1/IdentityIdentity+sequential/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/dropout_1/Identity“
)sequential/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02+
)sequential/conv2d_4/Conv2D/ReadVariableOpА
sequential/conv2d_4/Conv2DConv2D&sequential/dropout_1/Identity:output:01sequential/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential/conv2d_4/Conv2D…
*sequential/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*sequential/conv2d_4/BiasAdd/ReadVariableOpў
sequential/conv2d_4/BiasAddBiasAdd#sequential/conv2d_4/Conv2D:output:02sequential/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_4/BiasAddЭ
sequential/conv2d_4/ReluRelu$sequential/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_4/ReluЎ
/sequential/batch_normalization_4/ReadVariableOpReadVariableOp8sequential_batch_normalization_4_readvariableop_resource*
_output_shapes	
:А*
dtype021
/sequential/batch_normalization_4/ReadVariableOpё
1sequential/batch_normalization_4/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype023
1sequential/batch_normalization_4/ReadVariableOp_1Л
@sequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02B
@sequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOpС
Bsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02D
Bsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1µ
1sequential/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_4/Relu:activations:07sequential/batch_normalization_4/ReadVariableOp:value:09sequential/batch_normalization_4/ReadVariableOp_1:value:0Hsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_4/FusedBatchNormV3”
)sequential/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02+
)sequential/conv2d_5/Conv2D/ReadVariableOpП
sequential/conv2d_5/Conv2DConv2D5sequential/batch_normalization_4/FusedBatchNormV3:y:01sequential/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
sequential/conv2d_5/Conv2D…
*sequential/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*sequential/conv2d_5/BiasAdd/ReadVariableOpў
sequential/conv2d_5/BiasAddBiasAdd#sequential/conv2d_5/Conv2D:output:02sequential/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_5/BiasAddЭ
sequential/conv2d_5/ReluRelu$sequential/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_5/ReluЎ
/sequential/batch_normalization_5/ReadVariableOpReadVariableOp8sequential_batch_normalization_5_readvariableop_resource*
_output_shapes	
:А*
dtype021
/sequential/batch_normalization_5/ReadVariableOpё
1sequential/batch_normalization_5/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:А*
dtype023
1sequential/batch_normalization_5/ReadVariableOp_1Л
@sequential/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02B
@sequential/batch_normalization_5/FusedBatchNormV3/ReadVariableOpС
Bsequential/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02D
Bsequential/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1µ
1sequential/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3&sequential/conv2d_5/Relu:activations:07sequential/batch_normalization_5/ReadVariableOp:value:09sequential/batch_normalization_5/ReadVariableOp_1:value:0Hsequential/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 23
1sequential/batch_normalization_5/FusedBatchNormV3ш
"sequential/max_pooling2d_2/MaxPoolMaxPool5sequential/batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2$
"sequential/max_pooling2d_2/MaxPool≤
sequential/dropout_2/IdentityIdentity+sequential/max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/dropout_2/IdentityЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/ConstЅ
sequential/flatten/ReshapeReshape&sequential/dropout_2/Identity:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape¬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02(
&sequential/dense/MatMul/ReadVariableOpƒ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dense/MatMulј
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp∆
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dense/BiasAddМ
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dense/Reluц
9sequential/batch_normalization_6/batchnorm/ReadVariableOpReadVariableOpBsequential_batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9sequential/batch_normalization_6/batchnorm/ReadVariableOp©
0sequential/batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0sequential/batch_normalization_6/batchnorm/add/yН
.sequential/batch_normalization_6/batchnorm/addAddV2Asequential/batch_normalization_6/batchnorm/ReadVariableOp:value:09sequential/batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_6/batchnorm/add«
0sequential/batch_normalization_6/batchnorm/RsqrtRsqrt2sequential/batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А22
0sequential/batch_normalization_6/batchnorm/RsqrtВ
=sequential/batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOpFsequential_batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02?
=sequential/batch_normalization_6/batchnorm/mul/ReadVariableOpК
.sequential/batch_normalization_6/batchnorm/mulMul4sequential/batch_normalization_6/batchnorm/Rsqrt:y:0Esequential/batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_6/batchnorm/mulч
0sequential/batch_normalization_6/batchnorm/mul_1Mul#sequential/dense/Relu:activations:02sequential/batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А22
0sequential/batch_normalization_6/batchnorm/mul_1ь
;sequential/batch_normalization_6/batchnorm/ReadVariableOp_1ReadVariableOpDsequential_batch_normalization_6_batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02=
;sequential/batch_normalization_6/batchnorm/ReadVariableOp_1К
0sequential/batch_normalization_6/batchnorm/mul_2MulCsequential/batch_normalization_6/batchnorm/ReadVariableOp_1:value:02sequential/batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А22
0sequential/batch_normalization_6/batchnorm/mul_2ь
;sequential/batch_normalization_6/batchnorm/ReadVariableOp_2ReadVariableOpDsequential_batch_normalization_6_batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02=
;sequential/batch_normalization_6/batchnorm/ReadVariableOp_2И
.sequential/batch_normalization_6/batchnorm/subSubCsequential/batch_normalization_6/batchnorm/ReadVariableOp_2:value:04sequential/batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А20
.sequential/batch_normalization_6/batchnorm/subК
0sequential/batch_normalization_6/batchnorm/add_1AddV24sequential/batch_normalization_6/batchnorm/mul_1:z:02sequential/batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А22
0sequential/batch_normalization_6/batchnorm/add_1≥
sequential/dropout_3/IdentityIdentity4sequential/batch_normalization_6/batchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/dropout_3/Identity«
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpћ
sequential/dense_1/MatMulMatMul&sequential/dropout_3/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_1/Softmaxx
IdentityIdentity$sequential/dense_1/Softmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  :::::::::::::::::::::::::::::::::::::::::::::Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_11:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
р
ђ
D__inference_dense_1_layer_call_and_return_conditional_losses_1639591

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
≠
™
7__inference_batch_normalization_2_layer_call_fn_1641489

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16390912
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ќ
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_1642208

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ш
~
)__inference_dense_1_layer_call_fn_1642238

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall“
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16395912
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Р
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1637916

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ :::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≥

≠
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1638247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingSAME*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
ReluА
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Д
Ъ
,__inference_sequential_layer_call_fn_1640140
input_11
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

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42*8
Tin1
/2-*
Tout
2*'
_output_shapes
:€€€€€€€€€
*N
_read_only_resource_inputs0
.,	
 !"#$%&'()*+,**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_16400492
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:€€€€€€€€€  
"
_user_specified_name
input_11:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
ц
|
'__inference_dense_layer_call_fn_1642073

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCall—
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16394992
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ю
Л
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1638532

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1б
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Г
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
С
™
7__inference_batch_normalization_6_layer_call_fn_1642178

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_16387992
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Т
Л
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641563

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Њ*
Ќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1642145

inputs
assignmovingavg_1642120
assignmovingavg_1_1642126)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradient•
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≥
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/1642120*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1642120*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp≈
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/1642120*
_output_shapes	
:А2
AssignMovingAvg/subЉ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/1642120*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1642120AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/1642120*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp•
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/1642126*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1642126*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1642126*
_output_shapes	
:А2
AssignMovingAvg_1/sub∆
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1642126*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1642126AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/1642126*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1ґ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ћ
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1639456

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Н
M
1__inference_max_pooling2d_1_layer_call_fn_1638395

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_16383892
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≠
™
7__inference_batch_normalization_3_layer_call_fn_1641651

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16391802
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
†$
ў
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1639180

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Њ
`
D__inference_flatten_layer_call_and_return_conditional_losses_1642048

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
н
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1639461

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
и$
ў
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1638033

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ЫЅ
є
#__inference__traced_restore_1642541
file_prefix%
!assignvariableop_conv2d_36_kernel%
!assignvariableop_1_conv2d_36_bias3
/assignvariableop_2_batch_normalization_37_gamma2
.assignvariableop_3_batch_normalization_37_beta9
5assignvariableop_4_batch_normalization_37_moving_mean=
9assignvariableop_5_batch_normalization_37_moving_variance)
%assignvariableop_6_conv2d_1_31_kernel'
#assignvariableop_7_conv2d_1_31_bias5
1assignvariableop_8_batch_normalization_1_31_gamma4
0assignvariableop_9_batch_normalization_1_31_beta<
8assignvariableop_10_batch_normalization_1_31_moving_mean@
<assignvariableop_11_batch_normalization_1_31_moving_variance*
&assignvariableop_12_conv2d_2_31_kernel(
$assignvariableop_13_conv2d_2_31_bias6
2assignvariableop_14_batch_normalization_2_31_gamma5
1assignvariableop_15_batch_normalization_2_31_beta<
8assignvariableop_16_batch_normalization_2_31_moving_mean@
<assignvariableop_17_batch_normalization_2_31_moving_variance*
&assignvariableop_18_conv2d_3_31_kernel(
$assignvariableop_19_conv2d_3_31_bias6
2assignvariableop_20_batch_normalization_3_31_gamma5
1assignvariableop_21_batch_normalization_3_31_beta<
8assignvariableop_22_batch_normalization_3_31_moving_mean@
<assignvariableop_23_batch_normalization_3_31_moving_variance*
&assignvariableop_24_conv2d_4_31_kernel(
$assignvariableop_25_conv2d_4_31_bias6
2assignvariableop_26_batch_normalization_4_31_gamma5
1assignvariableop_27_batch_normalization_4_31_beta<
8assignvariableop_28_batch_normalization_4_31_moving_mean@
<assignvariableop_29_batch_normalization_4_31_moving_variance*
&assignvariableop_30_conv2d_5_31_kernel(
$assignvariableop_31_conv2d_5_31_bias6
2assignvariableop_32_batch_normalization_5_31_gamma5
1assignvariableop_33_batch_normalization_5_31_beta<
8assignvariableop_34_batch_normalization_5_31_moving_mean@
<assignvariableop_35_batch_normalization_5_31_moving_variance'
#assignvariableop_36_dense_32_kernel%
!assignvariableop_37_dense_32_bias6
2assignvariableop_38_batch_normalization_6_31_gamma5
1assignvariableop_39_batch_normalization_6_31_beta<
8assignvariableop_40_batch_normalization_6_31_moving_mean@
<assignvariableop_41_batch_normalization_6_31_moving_variance)
%assignvariableop_42_dense_1_31_kernel'
#assignvariableop_43_dense_1_31_bias
identity_45ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*А
valueцBу,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesж
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesК
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*∆
_output_shapes≥
∞::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityС
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_36_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ч
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_36_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2•
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_37_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_37_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ђ
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_37_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5ѓ
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_37_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ы
AssignVariableOp_6AssignVariableOp%assignvariableop_6_conv2d_1_31_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Щ
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2d_1_31_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8І
AssignVariableOp_8AssignVariableOp1assignvariableop_8_batch_normalization_1_31_gammaIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9¶
AssignVariableOp_9AssignVariableOp0assignvariableop_9_batch_normalization_1_31_betaIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10±
AssignVariableOp_10AssignVariableOp8assignvariableop_10_batch_normalization_1_31_moving_meanIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11µ
AssignVariableOp_11AssignVariableOp<assignvariableop_11_batch_normalization_1_31_moving_varianceIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Я
AssignVariableOp_12AssignVariableOp&assignvariableop_12_conv2d_2_31_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Э
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2d_2_31_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp2assignvariableop_14_batch_normalization_2_31_gammaIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15™
AssignVariableOp_15AssignVariableOp1assignvariableop_15_batch_normalization_2_31_betaIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp8assignvariableop_16_batch_normalization_2_31_moving_meanIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17µ
AssignVariableOp_17AssignVariableOp<assignvariableop_17_batch_normalization_2_31_moving_varianceIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Я
AssignVariableOp_18AssignVariableOp&assignvariableop_18_conv2d_3_31_kernelIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Э
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv2d_3_31_biasIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Ђ
AssignVariableOp_20AssignVariableOp2assignvariableop_20_batch_normalization_3_31_gammaIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21™
AssignVariableOp_21AssignVariableOp1assignvariableop_21_batch_normalization_3_31_betaIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp8assignvariableop_22_batch_normalization_3_31_moving_meanIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23µ
AssignVariableOp_23AssignVariableOp<assignvariableop_23_batch_normalization_3_31_moving_varianceIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Я
AssignVariableOp_24AssignVariableOp&assignvariableop_24_conv2d_4_31_kernelIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Э
AssignVariableOp_25AssignVariableOp$assignvariableop_25_conv2d_4_31_biasIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Ђ
AssignVariableOp_26AssignVariableOp2assignvariableop_26_batch_normalization_4_31_gammaIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27™
AssignVariableOp_27AssignVariableOp1assignvariableop_27_batch_normalization_4_31_betaIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp8assignvariableop_28_batch_normalization_4_31_moving_meanIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29µ
AssignVariableOp_29AssignVariableOp<assignvariableop_29_batch_normalization_4_31_moving_varianceIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30Я
AssignVariableOp_30AssignVariableOp&assignvariableop_30_conv2d_5_31_kernelIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31Э
AssignVariableOp_31AssignVariableOp$assignvariableop_31_conv2d_5_31_biasIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32Ђ
AssignVariableOp_32AssignVariableOp2assignvariableop_32_batch_normalization_5_31_gammaIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33™
AssignVariableOp_33AssignVariableOp1assignvariableop_33_batch_normalization_5_31_betaIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp8assignvariableop_34_batch_normalization_5_31_moving_meanIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35µ
AssignVariableOp_35AssignVariableOp<assignvariableop_35_batch_normalization_5_31_moving_varianceIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36Ь
AssignVariableOp_36AssignVariableOp#assignvariableop_36_dense_32_kernelIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Ъ
AssignVariableOp_37AssignVariableOp!assignvariableop_37_dense_32_biasIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Ђ
AssignVariableOp_38AssignVariableOp2assignvariableop_38_batch_normalization_6_31_gammaIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39™
AssignVariableOp_39AssignVariableOp1assignvariableop_39_batch_normalization_6_31_betaIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40±
AssignVariableOp_40AssignVariableOp8assignvariableop_40_batch_normalization_6_31_moving_meanIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41µ
AssignVariableOp_41AssignVariableOp<assignvariableop_41_batch_normalization_6_31_moving_varianceIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42Ю
AssignVariableOp_42AssignVariableOp%assignvariableop_42_dense_1_31_kernelIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43Ь
AssignVariableOp_43AssignVariableOp#assignvariableop_43_dense_1_31_biasIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¶
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44≥
Identity_45IdentityIdentity_44:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_45"#
identity_45Identity_45:output:0*«
_input_shapesµ
≤: ::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
ж$
„
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1637885

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
й
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_1639252

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
К
Х
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1642165

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИУ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yЙ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1Щ
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_1Ж
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2Щ
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOp_2Д
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А:::::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ћ
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_1642027

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
†$
ў
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1639091

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ф
G
+__inference_dropout_1_layer_call_fn_1641691

inputs
identity™
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16392522
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€@:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
В
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1638389

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ѓ
™
7__inference_batch_normalization_2_layer_call_fn_1641502

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16391092
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
≤$
ў
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641971

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Љ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpњ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ђ
®
5__inference_batch_normalization_layer_call_fn_1641151

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16389002
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
и$
ў
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1638341

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
цѕ
О
G__inference_sequential_layer_call_and_return_conditional_losses_1640634

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource1
-batch_normalization_6_assignmovingavg_16405943
/batch_normalization_6_assignmovingavg_1_1640600?
;batch_normalization_6_batchnorm_mul_readvariableop_resource;
7batch_normalization_6_batchnorm_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityИҐ7batch_normalization/AssignMovingAvg/AssignSubVariableOpҐ9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpҐ9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpҐ;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpЄ
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d/Relu∞
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpґ
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpй
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1¬
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization/Constм
)batch_normalization/AssignMovingAvg/sub/xConst*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)batch_normalization/AssignMovingAvg/sub/x£
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0"batch_normalization/Const:output:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/subб
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp¬
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg/sub_1Ђ
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/mul”
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpт
+batch_normalization/AssignMovingAvg_1/sub/xConst*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization/AssignMovingAvg_1/sub/xЂ
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0"batch_normalization/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/subз
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOpќ
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization/AssignMovingAvg_1/sub_1µ
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/mulб
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpа
conv2d_1/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   *
paddingSAME*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€   2
conv2d_1/Reluґ
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpЉ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1й
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1–
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_1/Constт
+batch_normalization_1/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_1/AssignMovingAvg/sub/x≠
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subз
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg/sub_1µ
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mulб
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_1/AssignMovingAvg_1/sub/xµ
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0$batch_normalization_1/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/subн
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2/
-batch_normalization_1/AssignMovingAvg_1/sub_1њ
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/mulп
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp“
max_pooling2d/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?2
dropout/dropout/ConstЂ
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape‘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>2 
dropout/dropout/GreaterEqual/yж
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/dropout/GreaterEqualЯ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€ 2
dropout/dropout/CastҐ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
dropout/dropout/Mul_1∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp—
conv2d_2/Conv2DConv2Ddropout/dropout/Mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/Reluґ
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOpЉ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1й
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1–
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_2/Constт
+batch_normalization_2/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_2/AssignMovingAvg/sub/x≠
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/subз
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg/sub_1µ
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mulб
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_2/AssignMovingAvg_1/sub/xµ
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0$batch_normalization_2/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/subн
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2/
-batch_normalization_2/AssignMovingAvg_1/sub_1њ
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulп
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp∞
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpв
conv2d_3/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingSAME*
strides
2
conv2d_3/Conv2DІ
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOpђ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_3/Reluґ
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOpЉ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1й
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpп
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1–
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_3/Constт
+batch_normalization_3/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_3/AssignMovingAvg/sub/x≠
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0$batch_normalization_3/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/subз
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOpћ
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg/sub_1µ
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/mulб
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_3/AssignMovingAvg_1/sub/xµ
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0$batch_normalization_3/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/subн
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpЎ
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2/
-batch_normalization_3/AssignMovingAvg_1/sub_1њ
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/mulп
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp÷
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nџґ?2
dropout_1/dropout/Const≥
dropout_1/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/MulВ
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeЏ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2"
 dropout_1/dropout/GreaterEqual/yо
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2 
dropout_1/dropout/GreaterEqual•
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/Cast™
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€@2
dropout_1/dropout/Mul_1±
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_4/Conv2D/ReadVariableOp‘
conv2d_4/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_4/Conv2D®
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp≠
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_4/BiasAdd|
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_4/ReluЈ
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_4/ReadVariableOpљ
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_4/ReadVariableOp_1к
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpр
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1’
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2(
&batch_normalization_4/FusedBatchNormV3
batch_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_4/Constт
+batch_normalization_4/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_4/AssignMovingAvg/sub/x≠
)batch_normalization_4/AssignMovingAvg/subSub4batch_normalization_4/AssignMovingAvg/sub/x:output:0$batch_normalization_4/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/subи
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOpЌ
+batch_normalization_4/AssignMovingAvg/sub_1Sub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_4/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg/sub_1ґ
)batch_normalization_4/AssignMovingAvg/mulMul/batch_normalization_4/AssignMovingAvg/sub_1:z:0-batch_normalization_4/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2+
)batch_normalization_4/AssignMovingAvg/mulб
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp6^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_4/AssignMovingAvg_1/sub/xµ
+batch_normalization_4/AssignMovingAvg_1/subSub6batch_normalization_4/AssignMovingAvg_1/sub/x:output:0$batch_normalization_4/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/subо
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpў
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_4/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2/
-batch_normalization_4/AssignMovingAvg_1/sub_1ј
+batch_normalization_4/AssignMovingAvg_1/mulMul1batch_normalization_4/AssignMovingAvg_1/sub_1:z:0/batch_normalization_4/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2-
+batch_normalization_4/AssignMovingAvg_1/mulп
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp≤
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02 
conv2d_5/Conv2D/ReadVariableOpг
conv2d_5/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
2
conv2d_5/Conv2D®
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp≠
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_5/BiasAdd|
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_5/ReluЈ
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:А*
dtype02&
$batch_normalization_5/ReadVariableOpљ
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:А*
dtype02(
&batch_normalization_5/ReadVariableOp_1к
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpр
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1’
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2(
&batch_normalization_5/FusedBatchNormV3
batch_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
batch_normalization_5/Constт
+batch_normalization_5/AssignMovingAvg/sub/xConst*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2-
+batch_normalization_5/AssignMovingAvg/sub/x≠
)batch_normalization_5/AssignMovingAvg/subSub4batch_normalization_5/AssignMovingAvg/sub/x:output:0$batch_normalization_5/Const:output:0*
T0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/subи
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOpЌ
+batch_normalization_5/AssignMovingAvg/sub_1Sub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_5/FusedBatchNormV3:batch_mean:0*
T0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2-
+batch_normalization_5/AssignMovingAvg/sub_1ґ
)batch_normalization_5/AssignMovingAvg/mulMul/batch_normalization_5/AssignMovingAvg/sub_1:z:0-batch_normalization_5/AssignMovingAvg/sub:z:0*
T0*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2+
)batch_normalization_5/AssignMovingAvg/mulб
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp6^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpш
-batch_normalization_5/AssignMovingAvg_1/sub/xConst*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2/
-batch_normalization_5/AssignMovingAvg_1/sub/xµ
+batch_normalization_5/AssignMovingAvg_1/subSub6batch_normalization_5/AssignMovingAvg_1/sub/x:output:0$batch_normalization_5/Const:output:0*
T0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/subо
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpў
-batch_normalization_5/AssignMovingAvg_1/sub_1Sub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_5/FusedBatchNormV3:batch_variance:0*
T0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2/
-batch_normalization_5/AssignMovingAvg_1/sub_1ј
+batch_normalization_5/AssignMovingAvg_1/mulMul1batch_normalization_5/AssignMovingAvg_1/sub_1:z:0/batch_normalization_5/AssignMovingAvg_1/sub:z:0*
T0*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2-
+batch_normalization_5/AssignMovingAvg_1/mulп
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp„
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU’?2
dropout_2/dropout/Constі
dropout_2/dropout/MulMul max_pooling2d_2/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_2/dropout/MulВ
dropout_2/dropout/ShapeShape max_pooling2d_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeџ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ>2"
 dropout_2/dropout/GreaterEqual/yп
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
dropout_2/dropout/GreaterEqual¶
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€А2
dropout_2/dropout/CastЂ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dropout_2/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstХ
flatten/ReshapeReshapedropout_2/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

dense/Reluґ
4batch_normalization_6/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_6/moments/mean/reduction_indicesд
"batch_normalization_6/moments/meanMeandense/Relu:activations:0=batch_normalization_6/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2$
"batch_normalization_6/moments/meanњ
*batch_normalization_6/moments/StopGradientStopGradient+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes
:	А2,
*batch_normalization_6/moments/StopGradientщ
/batch_normalization_6/moments/SquaredDifferenceSquaredDifferencedense/Relu:activations:03batch_normalization_6/moments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А21
/batch_normalization_6/moments/SquaredDifferenceЊ
8batch_normalization_6/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_6/moments/variance/reduction_indicesЛ
&batch_normalization_6/moments/varianceMean3batch_normalization_6/moments/SquaredDifference:z:0Abatch_normalization_6/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2(
&batch_normalization_6/moments/variance√
%batch_normalization_6/moments/SqueezeSqueeze+batch_normalization_6/moments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2'
%batch_normalization_6/moments/SqueezeЋ
'batch_normalization_6/moments/Squeeze_1Squeeze/batch_normalization_6/moments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2)
'batch_normalization_6/moments/Squeeze_1б
+batch_normalization_6/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/1640594*
_output_shapes
: *
dtype0*
valueB
 *
„#<2-
+batch_normalization_6/AssignMovingAvg/decay„
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_6_assignmovingavg_1640594*
_output_shapes	
:А*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp≥
)batch_normalization_6/AssignMovingAvg/subSub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_6/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/1640594*
_output_shapes	
:А2+
)batch_normalization_6/AssignMovingAvg/sub™
)batch_normalization_6/AssignMovingAvg/mulMul-batch_normalization_6/AssignMovingAvg/sub:z:04batch_normalization_6/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/1640594*
_output_shapes	
:А2+
)batch_normalization_6/AssignMovingAvg/mulЗ
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_6_assignmovingavg_1640594-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg/1640594*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpз
-batch_normalization_6/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/1640600*
_output_shapes
: *
dtype0*
valueB
 *
„#<2/
-batch_normalization_6/AssignMovingAvg_1/decayЁ
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_6_assignmovingavg_1_1640600*
_output_shapes	
:А*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpљ
+batch_normalization_6/AssignMovingAvg_1/subSub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_6/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/1640600*
_output_shapes	
:А2-
+batch_normalization_6/AssignMovingAvg_1/subі
+batch_normalization_6/AssignMovingAvg_1/mulMul/batch_normalization_6/AssignMovingAvg_1/sub:z:06batch_normalization_6/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/1640600*
_output_shapes	
:А2-
+batch_normalization_6/AssignMovingAvg_1/mulУ
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_6_assignmovingavg_1_1640600/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_6/AssignMovingAvg_1/1640600*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_6/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_6/batchnorm/add/yџ
#batch_normalization_6/batchnorm/addAddV20batch_normalization_6/moments/Squeeze_1:output:0.batch_normalization_6/batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/add¶
%batch_normalization_6/batchnorm/RsqrtRsqrt'batch_normalization_6/batchnorm/add:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/Rsqrtб
2batch_normalization_6/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_6_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype024
2batch_normalization_6/batchnorm/mul/ReadVariableOpё
#batch_normalization_6/batchnorm/mulMul)batch_normalization_6/batchnorm/Rsqrt:y:0:batch_normalization_6/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/mulЋ
%batch_normalization_6/batchnorm/mul_1Muldense/Relu:activations:0'batch_normalization_6/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_6/batchnorm/mul_1‘
%batch_normalization_6/batchnorm/mul_2Mul.batch_normalization_6/moments/Squeeze:output:0'batch_normalization_6/batchnorm/mul:z:0*
T0*
_output_shapes	
:А2'
%batch_normalization_6/batchnorm/mul_2’
.batch_normalization_6/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_6_batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype020
.batch_normalization_6/batchnorm/ReadVariableOpЏ
#batch_normalization_6/batchnorm/subSub6batch_normalization_6/batchnorm/ReadVariableOp:value:0)batch_normalization_6/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2%
#batch_normalization_6/batchnorm/subё
%batch_normalization_6/batchnorm/add_1AddV2)batch_normalization_6/batchnorm/mul_1:z:0'batch_normalization_6/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%batch_normalization_6/batchnorm/add_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Constµ
dropout_3/dropout/MulMul)batch_normalization_6/batchnorm/add_1:z:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_3/dropout/MulЛ
dropout_3/dropout/ShapeShape)batch_normalization_6/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape”
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yз
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
dropout_3/dropout/GreaterEqualЮ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:€€€€€€€€€А2
dropout_3/dropout/Cast£
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dropout_3/dropout/Mul_1¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	А
*
dtype02
dense_1/MatMul/ReadVariableOp†
dense_1/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_1/Softmaxњ
IdentityIdentitydense_1/Softmax:softmax:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
а

*__inference_conv2d_2_layer_call_fn_1638109

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_16380992
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ѓ
™
7__inference_batch_normalization_3_layer_call_fn_1641664

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16391982
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш
G
+__inference_dropout_2_layer_call_fn_1642042

inputs
identityЂ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_16394612
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Њ*
Ќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1638799

inputs
assignmovingavg_1638774
assignmovingavg_1_1638780)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesР
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	А2
moments/StopGradient•
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices≥
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	А*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:А*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/1638774*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg/decayХ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_1638774*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOp≈
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/1638774*
_output_shapes	
:А2
AssignMovingAvg/subЉ
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/1638774*
_output_shapes	
:А2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1638774AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/1638774*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp•
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/1638780*
_output_shapes
: *
dtype0*
valueB
 *
„#<2
AssignMovingAvg_1/decayЫ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_1638780*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1638780*
_output_shapes	
:А2
AssignMovingAvg_1/sub∆
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/1638780*
_output_shapes	
:А2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_1638780AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/1638780*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yГ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:А2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:А2
batchnorm/RsqrtЯ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/mul/ReadVariableOpЖ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:А2
batchnorm/mul_2У
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:А*
dtype02
batchnorm/ReadVariableOpВ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:А2
batchnorm/subЖ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€А2
batchnorm/add_1ґ
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«
Й
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641125

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1 
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   :::::W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Љ

≠
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1638555

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИЧ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingSAME*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
ReluБ
IdentityIdentityRelu:activations:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:::j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ђz
ж
G__inference_sequential_layer_call_and_return_conditional_losses_1640049

inputs
conv2d_1639937
conv2d_1639939
batch_normalization_1639942
batch_normalization_1639944
batch_normalization_1639946
batch_normalization_1639948
conv2d_1_1639951
conv2d_1_1639953!
batch_normalization_1_1639956!
batch_normalization_1_1639958!
batch_normalization_1_1639960!
batch_normalization_1_1639962
conv2d_2_1639967
conv2d_2_1639969!
batch_normalization_2_1639972!
batch_normalization_2_1639974!
batch_normalization_2_1639976!
batch_normalization_2_1639978
conv2d_3_1639981
conv2d_3_1639983!
batch_normalization_3_1639986!
batch_normalization_3_1639988!
batch_normalization_3_1639990!
batch_normalization_3_1639992
conv2d_4_1639997
conv2d_4_1639999!
batch_normalization_4_1640002!
batch_normalization_4_1640004!
batch_normalization_4_1640006!
batch_normalization_4_1640008
conv2d_5_1640011
conv2d_5_1640013!
batch_normalization_5_1640016!
batch_normalization_5_1640018!
batch_normalization_5_1640020!
batch_normalization_5_1640022
dense_1640028
dense_1640030!
batch_normalization_6_1640033!
batch_normalization_6_1640035!
batch_normalization_6_1640037!
batch_normalization_6_1640039
dense_1_1640043
dense_1_1640045
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallу
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1639937conv2d_1639939*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_16377912 
conv2d/StatefulPartitionedCallУ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1639942batch_normalization_1639944batch_normalization_1639946batch_normalization_1639948*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_16389002-
+batch_normalization/StatefulPartitionedCallЂ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_1_1639951conv2d_1_1639953*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_16379392"
 conv2d_1/StatefulPartitionedCall£
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1639956batch_normalization_1_1639958batch_normalization_1_1639960batch_normalization_1_1639962*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€   *&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_16389892/
-batch_normalization_1/StatefulPartitionedCallъ
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_16380812
max_pooling2d/PartitionedCallЎ
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_16390432
dropout/PartitionedCallЧ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_2_1639967conv2d_2_1639969*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_16380992"
 conv2d_2/StatefulPartitionedCall£
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1639972batch_normalization_2_1639974batch_normalization_2_1639976batch_normalization_2_1639978*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_16391092/
-batch_normalization_2/StatefulPartitionedCall≠
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv2d_3_1639981conv2d_3_1639983*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_16382472"
 conv2d_3/StatefulPartitionedCall£
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1639986batch_normalization_3_1639988batch_normalization_3_1639990batch_normalization_3_1639992*
Tin	
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_16391982/
-batch_normalization_3/StatefulPartitionedCallА
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_16383892!
max_pooling2d_1/PartitionedCallа
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_16392522
dropout_1/PartitionedCallЪ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_4_1639997conv2d_4_1639999*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_16384072"
 conv2d_4/StatefulPartitionedCall§
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_1640002batch_normalization_4_1640004batch_normalization_4_1640006batch_normalization_4_1640008*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16393182/
-batch_normalization_4/StatefulPartitionedCallЃ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0conv2d_5_1640011conv2d_5_1640013*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16385552"
 conv2d_5/StatefulPartitionedCall§
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_1640016batch_normalization_5_1640018batch_normalization_5_1640020batch_normalization_5_1640022*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_16394072/
-batch_normalization_5/StatefulPartitionedCallБ
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_16386972!
max_pooling2d_2/PartitionedCallб
dropout_2/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_16394612
dropout_2/PartitionedCallЌ
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_16394802
flatten/PartitionedCallБ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1640028dense_1640030*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_16394992
dense/StatefulPartitionedCallЩ
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0batch_normalization_6_1640033batch_normalization_6_1640035batch_normalization_6_1640037batch_normalization_6_1640039*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_16388322/
-batch_normalization_6/StatefulPartitionedCallз
dropout_3/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_16395672
dropout_3/PartitionedCallМ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_1_1640043dense_1_1640045*
Tin
2*
Tout
2*'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_16395912!
dense_1/StatefulPartitionedCall№
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*а
_input_shapesќ
Ћ:€€€€€€€€€  ::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: 
У
™
7__inference_batch_normalization_6_layer_call_fn_1642191

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*(
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_16388322
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Т
Л
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641401

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1№
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3В
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:::::i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ъ$
ў
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641809

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub¶
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02 
AssignMovingAvg/ReadVariableOpя
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/sub_1»
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes	
:А2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subђ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02"
 AssignMovingAvg_1/ReadVariableOpл
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/sub_1“
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes	
:А2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp—
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ж$
„
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641032

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp–
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
’
Л
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641989

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1®
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpЃ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А:::::X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
†$
ў
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641269

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ю$
„
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1638882

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИҐ#AssignMovingAvg/AssignSubVariableOpҐ%AssignMovingAvg_1/AssignSubVariableOpt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1І
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp≠
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ј
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€   : : : : :*
epsilon%oГ:2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *§p}?2
Const∞
AssignMovingAvg/sub/xConst*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg/sub/xњ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const:output:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub•
AssignMovingAvg/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpё
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/sub_1«
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
: 2
AssignMovingAvg/mul«
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp(fusedbatchnormv3_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpґ
AssignMovingAvg_1/sub/xConst*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: *
dtype0*
valueB
 *  А?2
AssignMovingAvg_1/sub/x«
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const:output:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/subЂ
 AssignMovingAvg_1/ReadVariableOpReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/sub_1—
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
: 2
AssignMovingAvg_1/mul’
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp*fusedbatchnormv3_readvariableop_1_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЊ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*/
_output_shapes
:€€€€€€€€€   2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
д

*__inference_conv2d_5_layer_call_fn_1638565

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_16385552
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Й
K
/__inference_max_pooling2d_layer_call_fn_1638087

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_16380812
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
н
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_1642032

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€А2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
±
™
7__inference_batch_normalization_4_layer_call_fn_1641765

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_16393002
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:€€€€€€€€€А::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*і
serving_default†
E
input_119
serving_default_input_11:0€€€€€€€€€  ;
dense_10
StatefulPartitionedCall:0€€€€€€€€€
tensorflow/serving/predict:РС
ьЈ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer_with_weights-10
layer-14
layer_with_weights-11
layer-15
layer-16
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer-21
layer_with_weights-14
layer-22
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
•__call__
+¶&call_and_return_all_conditional_losses
І_default_save_signature"к∞
_tf_keras_sequential ∞{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.01, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
ь


kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
®__call__
+©&call_and_return_all_conditional_losses"’	
_tf_keras_layerї	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "stateful": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
Х	
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)	variables
*trainable_variables
+regularization_losses
,	keras_api
™__call__
+Ђ&call_and_return_all_conditional_losses"њ
_tf_keras_layer•{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
Б


-kernel
.bias
/	variables
0trainable_variables
1regularization_losses
2	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses"Џ
_tf_keras_layerј{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
Щ	
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8	variables
9trainable_variables
:regularization_losses
;	keras_api
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
Џ
<	variables
=trainable_variables
>regularization_losses
?	keras_api
∞__call__
+±&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ј
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses"ѓ
_tf_keras_layerХ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Б


Dkernel
Ebias
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
і__call__
+µ&call_and_return_all_conditional_losses"Џ
_tf_keras_layerј{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
Щ	
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
ґ__call__
+Ј&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
Б


Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"Џ
_tf_keras_layerј{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
Щ	
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance
^	variables
_trainable_variables
`regularization_losses
a	keras_api
Ї__call__
+ї&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
ё
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
Љ__call__
+љ&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer≥{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ƒ
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
Њ__call__
+њ&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
А


jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses"ў
_tf_keras_layerњ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
Щ	
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
¬__call__
+√&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
В


ykernel
zbias
{	variables
|trainable_variables
}regularization_losses
~	keras_api
ƒ__call__
+≈&call_and_return_all_conditional_losses"џ
_tf_keras_layerЅ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
°	
axis

Аgamma
	Бbeta
Вmoving_mean
Гmoving_variance
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
∆__call__
+«&call_and_return_all_conditional_losses"√
_tf_keras_layer©{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
в
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
»__call__
+…&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer≥{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
»
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
 __call__
+Ћ&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.4, "noise_shape": null, "seed": null}}
≈
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
ћ__call__
+Ќ&call_and_return_all_conditional_losses"∞
_tf_keras_layerЦ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
У
Фkernel
	Хbias
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses"ж
_tf_keras_layerћ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "uniform", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2048}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2048]}}
Ь	
	Ъaxis

Ыgamma
	Ьbeta
Эmoving_mean
Юmoving_variance
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
–__call__
+—&call_and_return_all_conditional_losses"љ
_tf_keras_layer£{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
»
£	variables
§trainable_variables
•regularization_losses
¶	keras_api
“__call__
+”&call_and_return_all_conditional_losses"≥
_tf_keras_layerЩ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
Џ
Іkernel
	®bias
©	variables
™trainable_variables
Ђregularization_losses
ђ	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"≠
_tf_keras_layerУ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
"
	optimizer
В
0
1
%2
&3
'4
(5
-6
.7
48
59
610
711
D12
E13
K14
L15
M16
N17
S18
T19
Z20
[21
\22
]23
j24
k25
q26
r27
s28
t29
y30
z31
А32
Б33
В34
Г35
Ф36
Х37
Ы38
Ь39
Э40
Ю41
І42
®43"
trackable_list_wrapper
О
0
1
%2
&3
-4
.5
46
57
D8
E9
K10
L11
S12
T13
Z14
[15
j16
k17
q18
r19
y20
z21
А22
Б23
Ф24
Х25
Ы26
Ь27
І28
®29"
trackable_list_wrapper
 "
trackable_list_wrapper
”
	variables
trainable_variables
regularization_losses
≠non_trainable_variables
Ѓlayers
 ѓlayer_regularization_losses
∞metrics
±layer_metrics
•__call__
І_default_save_signature
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
-
÷serving_default"
signature_map
*:( 2conv2d_36/kernel
: 2conv2d_36/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 	variables
!trainable_variables
"regularization_losses
≤layers
≥non_trainable_variables
 іlayer_regularization_losses
µmetrics
ґlayer_metrics
®__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:( 2batch_normalization_37/gamma
):' 2batch_normalization_37/beta
2:0  (2"batch_normalization_37/moving_mean
6:4  (2&batch_normalization_37/moving_variance
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
)	variables
*trainable_variables
+regularization_losses
Јlayers
Єnon_trainable_variables
 єlayer_regularization_losses
Їmetrics
їlayer_metrics
™__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
,:*  2conv2d_1_31/kernel
: 2conv2d_1_31/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
/	variables
0trainable_variables
1regularization_losses
Љlayers
љnon_trainable_variables
 Њlayer_regularization_losses
њmetrics
јlayer_metrics
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:* 2batch_normalization_1_31/gamma
+:) 2batch_normalization_1_31/beta
4:2  (2$batch_normalization_1_31/moving_mean
8:6  (2(batch_normalization_1_31/moving_variance
<
40
51
62
73"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
8	variables
9trainable_variables
:regularization_losses
Ѕlayers
¬non_trainable_variables
 √layer_regularization_losses
ƒmetrics
≈layer_metrics
Ѓ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
<	variables
=trainable_variables
>regularization_losses
∆layers
«non_trainable_variables
 »layer_regularization_losses
…metrics
 layer_metrics
∞__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
@	variables
Atrainable_variables
Bregularization_losses
Ћlayers
ћnon_trainable_variables
 Ќlayer_regularization_losses
ќmetrics
ѕlayer_metrics
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
,:* @2conv2d_2_31/kernel
:@2conv2d_2_31/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
F	variables
Gtrainable_variables
Hregularization_losses
–layers
—non_trainable_variables
 “layer_regularization_losses
”metrics
‘layer_metrics
і__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*@2batch_normalization_2_31/gamma
+:)@2batch_normalization_2_31/beta
4:2@ (2$batch_normalization_2_31/moving_mean
8:6@ (2(batch_normalization_2_31/moving_variance
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
O	variables
Ptrainable_variables
Qregularization_losses
’layers
÷non_trainable_variables
 „layer_regularization_losses
Ўmetrics
ўlayer_metrics
ґ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
,:*@@2conv2d_3_31/kernel
:@2conv2d_3_31/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
U	variables
Vtrainable_variables
Wregularization_losses
Џlayers
џnon_trainable_variables
 №layer_regularization_losses
Ёmetrics
ёlayer_metrics
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
,:*@2batch_normalization_3_31/gamma
+:)@2batch_normalization_3_31/beta
4:2@ (2$batch_normalization_3_31/moving_mean
8:6@ (2(batch_normalization_3_31/moving_variance
<
Z0
[1
\2
]3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
^	variables
_trainable_variables
`regularization_losses
яlayers
аnon_trainable_variables
 бlayer_regularization_losses
вmetrics
гlayer_metrics
Ї__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
b	variables
ctrainable_variables
dregularization_losses
дlayers
еnon_trainable_variables
 жlayer_regularization_losses
зmetrics
иlayer_metrics
Љ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
f	variables
gtrainable_variables
hregularization_losses
йlayers
кnon_trainable_variables
 лlayer_regularization_losses
мmetrics
нlayer_metrics
Њ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
-:+@А2conv2d_4_31/kernel
:А2conv2d_4_31/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
l	variables
mtrainable_variables
nregularization_losses
оlayers
пnon_trainable_variables
 рlayer_regularization_losses
сmetrics
тlayer_metrics
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
-:+А2batch_normalization_4_31/gamma
,:*А2batch_normalization_4_31/beta
5:3А (2$batch_normalization_4_31/moving_mean
9:7А (2(batch_normalization_4_31/moving_variance
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
u	variables
vtrainable_variables
wregularization_losses
уlayers
фnon_trainable_variables
 хlayer_regularization_losses
цmetrics
чlayer_metrics
¬__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
.:,АА2conv2d_5_31/kernel
:А2conv2d_5_31/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
{	variables
|trainable_variables
}regularization_losses
шlayers
щnon_trainable_variables
 ъlayer_regularization_losses
ыmetrics
ьlayer_metrics
ƒ__call__
+≈&call_and_return_all_conditional_losses
'≈"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
-:+А2batch_normalization_5_31/gamma
,:*А2batch_normalization_5_31/beta
5:3А (2$batch_normalization_5_31/moving_mean
9:7А (2(batch_normalization_5_31/moving_variance
@
А0
Б1
В2
Г3"
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Д	variables
Еtrainable_variables
Жregularization_losses
эlayers
юnon_trainable_variables
 €layer_regularization_losses
Аmetrics
Бlayer_metrics
∆__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
И	variables
Йtrainable_variables
Кregularization_losses
Вlayers
Гnon_trainable_variables
 Дlayer_regularization_losses
Еmetrics
Жlayer_metrics
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
М	variables
Нtrainable_variables
Оregularization_losses
Зlayers
Иnon_trainable_variables
 Йlayer_regularization_losses
Кmetrics
Лlayer_metrics
 __call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Р	variables
Сtrainable_variables
Тregularization_losses
Мlayers
Нnon_trainable_variables
 Оlayer_regularization_losses
Пmetrics
Рlayer_metrics
ћ__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
#:!
АА2dense_32/kernel
:А2dense_32/bias
0
Ф0
Х1"
trackable_list_wrapper
0
Ф0
Х1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ц	variables
Чtrainable_variables
Шregularization_losses
Сlayers
Тnon_trainable_variables
 Уlayer_regularization_losses
Фmetrics
Хlayer_metrics
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
-:+А2batch_normalization_6_31/gamma
,:*А2batch_normalization_6_31/beta
5:3А (2$batch_normalization_6_31/moving_mean
9:7А (2(batch_normalization_6_31/moving_variance
@
Ы0
Ь1
Э2
Ю3"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Я	variables
†trainable_variables
°regularization_losses
Цlayers
Чnon_trainable_variables
 Шlayer_regularization_losses
Щmetrics
Ъlayer_metrics
–__call__
+—&call_and_return_all_conditional_losses
'—"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
£	variables
§trainable_variables
•regularization_losses
Ыlayers
Ьnon_trainable_variables
 Эlayer_regularization_losses
Юmetrics
Яlayer_metrics
“__call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
_generic_user_object
$:"	А
2dense_1_31/kernel
:
2dense_1_31/bias
0
І0
®1"
trackable_list_wrapper
0
І0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
©	variables
™trainable_variables
Ђregularization_losses
†layers
°non_trainable_variables
 Ґlayer_regularization_losses
£metrics
§layer_metrics
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
К
'0
(1
62
73
M4
N5
\6
]7
s8
t9
В10
Г11
Э12
Ю13"
trackable_list_wrapper
ќ
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
11
12
13
14
15
16
17
18
19
20
21
22"
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
.
'0
(1"
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
.
60
71"
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
.
M0
N1"
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
.
\0
]1"
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
.
s0
t1"
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
0
В0
Г1"
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
0
Э0
Ю1"
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
ю2ы
,__inference_sequential_layer_call_fn_1640896
,__inference_sequential_layer_call_fn_1639932
,__inference_sequential_layer_call_fn_1640140
,__inference_sequential_layer_call_fn_1640989ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
G__inference_sequential_layer_call_and_return_conditional_losses_1640634
G__inference_sequential_layer_call_and_return_conditional_losses_1639723
G__inference_sequential_layer_call_and_return_conditional_losses_1640803
G__inference_sequential_layer_call_and_return_conditional_losses_1639608ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
й2ж
"__inference__wrapped_model_1637779њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ */Ґ,
*К'
input_11€€€€€€€€€  
З2Д
(__inference_conv2d_layer_call_fn_1637801„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ґ2Я
C__inference_conv2d_layer_call_and_return_conditional_losses_1637791„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ц2У
5__inference_batch_normalization_layer_call_fn_1641151
5__inference_batch_normalization_layer_call_fn_1641076
5__inference_batch_normalization_layer_call_fn_1641138
5__inference_batch_normalization_layer_call_fn_1641063і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
В2€
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641107
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641050
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641032
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641125і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Й2Ж
*__inference_conv2d_1_layer_call_fn_1637949„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
§2°
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1637939„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ю2Ы
7__inference_batch_normalization_1_layer_call_fn_1641238
7__inference_batch_normalization_1_layer_call_fn_1641313
7__inference_batch_normalization_1_layer_call_fn_1641300
7__inference_batch_normalization_1_layer_call_fn_1641225і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
К2З
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641194
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641212
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641287
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641269і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_layer_call_fn_1638087а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1638081а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Р2Н
)__inference_dropout_layer_call_fn_1641335
)__inference_dropout_layer_call_fn_1641340і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
∆2√
D__inference_dropout_layer_call_and_return_conditional_losses_1641325
D__inference_dropout_layer_call_and_return_conditional_losses_1641330і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Й2Ж
*__inference_conv2d_2_layer_call_fn_1638109„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
§2°
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1638099„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ю2Ы
7__inference_batch_normalization_2_layer_call_fn_1641414
7__inference_batch_normalization_2_layer_call_fn_1641502
7__inference_batch_normalization_2_layer_call_fn_1641427
7__inference_batch_normalization_2_layer_call_fn_1641489і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
К2З
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641383
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641476
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641401
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641458і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Й2Ж
*__inference_conv2d_3_layer_call_fn_1638257„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
§2°
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1638247„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ю2Ы
7__inference_batch_normalization_3_layer_call_fn_1641664
7__inference_batch_normalization_3_layer_call_fn_1641589
7__inference_batch_normalization_3_layer_call_fn_1641651
7__inference_batch_normalization_3_layer_call_fn_1641576і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
К2З
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641638
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641563
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641620
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641545і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Щ2Ц
1__inference_max_pooling2d_1_layer_call_fn_1638395а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
і2±
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1638389а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
+__inference_dropout_1_layer_call_fn_1641691
+__inference_dropout_1_layer_call_fn_1641686і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 2«
F__inference_dropout_1_layer_call_and_return_conditional_losses_1641681
F__inference_dropout_1_layer_call_and_return_conditional_losses_1641676і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Й2Ж
*__inference_conv2d_4_layer_call_fn_1638417„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
§2°
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1638407„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ю2Ы
7__inference_batch_normalization_4_layer_call_fn_1641778
7__inference_batch_normalization_4_layer_call_fn_1641840
7__inference_batch_normalization_4_layer_call_fn_1641765
7__inference_batch_normalization_4_layer_call_fn_1641853і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
К2З
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641752
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641734
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641809
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641827і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
К2З
*__inference_conv2d_5_layer_call_fn_1638565Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
•2Ґ
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1638555Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ю2Ы
7__inference_batch_normalization_5_layer_call_fn_1641927
7__inference_batch_normalization_5_layer_call_fn_1642015
7__inference_batch_normalization_5_layer_call_fn_1642002
7__inference_batch_normalization_5_layer_call_fn_1641940і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
К2З
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641896
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641914
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641989
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641971і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Щ2Ц
1__inference_max_pooling2d_2_layer_call_fn_1638703а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
і2±
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1638697а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ф2С
+__inference_dropout_2_layer_call_fn_1642042
+__inference_dropout_2_layer_call_fn_1642037і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 2«
F__inference_dropout_2_layer_call_and_return_conditional_losses_1642027
F__inference_dropout_2_layer_call_and_return_conditional_losses_1642032і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_flatten_layer_call_fn_1642053Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_flatten_layer_call_and_return_conditional_losses_1642048Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_layer_call_fn_1642073Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_layer_call_and_return_conditional_losses_1642064Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ђ2©
7__inference_batch_normalization_6_layer_call_fn_1642191
7__inference_batch_normalization_6_layer_call_fn_1642178і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1642165
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1642145і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ф2С
+__inference_dropout_3_layer_call_fn_1642213
+__inference_dropout_3_layer_call_fn_1642218і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 2«
F__inference_dropout_3_layer_call_and_return_conditional_losses_1642203
F__inference_dropout_3_layer_call_and_return_conditional_losses_1642208і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
”2–
)__inference_dense_1_layer_call_fn_1642238Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_1_layer_call_and_return_conditional_losses_1642229Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
5B3
%__inference_signature_wrapper_1640343input_11ѕ
"__inference__wrapped_model_1637779®8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЮЫЭЬІ®9Ґ6
/Ґ,
*К'
input_11€€€€€€€€€  
™ "1™.
,
dense_1!К
dense_1€€€€€€€€€
н
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641194Ц4567MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ н
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641212Ц4567MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ »
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641269r4567;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ »
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1641287r4567;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ ≈
7__inference_batch_normalization_1_layer_call_fn_1641225Й4567MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ≈
7__inference_batch_normalization_1_layer_call_fn_1641238Й4567MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ †
7__inference_batch_normalization_1_layer_call_fn_1641300e4567;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p
™ " К€€€€€€€€€   †
7__inference_batch_normalization_1_layer_call_fn_1641313e4567;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p 
™ " К€€€€€€€€€   н
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641383ЦKLMNMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ н
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641401ЦKLMNMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ »
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641458rKLMN;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ »
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1641476rKLMN;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ ≈
7__inference_batch_normalization_2_layer_call_fn_1641414ЙKLMNMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@≈
7__inference_batch_normalization_2_layer_call_fn_1641427ЙKLMNMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@†
7__inference_batch_normalization_2_layer_call_fn_1641489eKLMN;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ " К€€€€€€€€€@†
7__inference_batch_normalization_2_layer_call_fn_1641502eKLMN;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ " К€€€€€€€€€@н
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641545ЦZ[\]MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ н
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641563ЦZ[\]MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ »
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641620rZ[\];Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ »
R__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1641638rZ[\];Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ ≈
7__inference_batch_normalization_3_layer_call_fn_1641576ЙZ[\]MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@≈
7__inference_batch_normalization_3_layer_call_fn_1641589ЙZ[\]MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@†
7__inference_batch_normalization_3_layer_call_fn_1641651eZ[\];Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ " К€€€€€€€€€@†
7__inference_batch_normalization_3_layer_call_fn_1641664eZ[\];Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ " К€€€€€€€€€@ 
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641734tqrst<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ  
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641752tqrst<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ п
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641809ШqrstNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ п
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1641827ШqrstNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ Ґ
7__inference_batch_normalization_4_layer_call_fn_1641765gqrst<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€АҐ
7__inference_batch_normalization_4_layer_call_fn_1641778gqrst<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€А«
7__inference_batch_normalization_4_layer_call_fn_1641840ЛqrstNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А«
7__inference_batch_normalization_4_layer_call_fn_1641853ЛqrstNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Ау
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641896ЬАБВГNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ у
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641914ЬАБВГNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641971xАБВГ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ ќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1641989xАБВГ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Ћ
7__inference_batch_normalization_5_layer_call_fn_1641927ПАБВГNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЋ
7__inference_batch_normalization_5_layer_call_fn_1641940ПАБВГNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А¶
7__inference_batch_normalization_5_layer_call_fn_1642002kАБВГ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€А¶
7__inference_batch_normalization_5_layer_call_fn_1642015kАБВГ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€АЊ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1642145hЭЮЫЬ4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Њ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1642165hЮЫЭЬ4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ Ц
7__inference_batch_normalization_6_layer_call_fn_1642178[ЭЮЫЬ4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АЦ
7__inference_batch_normalization_6_layer_call_fn_1642191[ЮЫЭЬ4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€Ал
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641032Ц%&'(MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ л
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641050Ц%&'(MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ∆
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641107r%&'(;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ ∆
P__inference_batch_normalization_layer_call_and_return_conditional_losses_1641125r%&'(;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ √
5__inference_batch_normalization_layer_call_fn_1641063Й%&'(MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ √
5__inference_batch_normalization_layer_call_fn_1641076Й%&'(MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Ю
5__inference_batch_normalization_layer_call_fn_1641138e%&'(;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p
™ " К€€€€€€€€€   Ю
5__inference_batch_normalization_layer_call_fn_1641151e%&'(;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€   
p 
™ " К€€€€€€€€€   Џ
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1637939Р-.IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≤
*__inference_conv2d_1_layer_call_fn_1637949Г-.IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Џ
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1638099РDEIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≤
*__inference_conv2d_2_layer_call_fn_1638109ГDEIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Џ
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1638247РSTIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≤
*__inference_conv2d_3_layer_call_fn_1638257ГSTIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@џ
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1638407СjkIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ≥
*__inference_conv2d_4_layer_call_fn_1638417ДjkIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А№
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1638555ТyzJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ і
*__inference_conv2d_5_layer_call_fn_1638565ЕyzJҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЎ
C__inference_conv2d_layer_call_and_return_conditional_losses_1637791РIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ∞
(__inference_conv2d_layer_call_fn_1637801ГIҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ І
D__inference_dense_1_layer_call_and_return_conditional_losses_1642229_І®0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€

Ъ 
)__inference_dense_1_layer_call_fn_1642238RІ®0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€
¶
B__inference_dense_layer_call_and_return_conditional_losses_1642064`ФХ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ~
'__inference_dense_layer_call_fn_1642073SФХ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Аґ
F__inference_dropout_1_layer_call_and_return_conditional_losses_1641676l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ ґ
F__inference_dropout_1_layer_call_and_return_conditional_losses_1641681l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ О
+__inference_dropout_1_layer_call_fn_1641686_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p
™ " К€€€€€€€€€@О
+__inference_dropout_1_layer_call_fn_1641691_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@
p 
™ " К€€€€€€€€€@Є
F__inference_dropout_2_layer_call_and_return_conditional_losses_1642027n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Є
F__inference_dropout_2_layer_call_and_return_conditional_losses_1642032n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Р
+__inference_dropout_2_layer_call_fn_1642037a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€АР
+__inference_dropout_2_layer_call_fn_1642042a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€А®
F__inference_dropout_3_layer_call_and_return_conditional_losses_1642203^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "&Ґ#
К
0€€€€€€€€€А
Ъ ®
F__inference_dropout_3_layer_call_and_return_conditional_losses_1642208^4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
+__inference_dropout_3_layer_call_fn_1642213Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "К€€€€€€€€€АА
+__inference_dropout_3_layer_call_fn_1642218Q4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "К€€€€€€€€€Аі
D__inference_dropout_layer_call_and_return_conditional_losses_1641325l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 
p
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ і
D__inference_dropout_layer_call_and_return_conditional_losses_1641330l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 
p 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ М
)__inference_dropout_layer_call_fn_1641335_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 
p
™ " К€€€€€€€€€ М
)__inference_dropout_layer_call_fn_1641340_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€ 
p 
™ " К€€€€€€€€€ ™
D__inference_flatten_layer_call_and_return_conditional_losses_1642048b8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€А
Ъ В
)__inference_flatten_layer_call_fn_1642053U8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "К€€€€€€€€€Ап
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1638389ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ «
1__inference_max_pooling2d_1_layer_call_fn_1638395СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€п
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1638697ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ «
1__inference_max_pooling2d_2_layer_call_fn_1638703СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1638081ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_layer_call_fn_1638087СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€р
G__inference_sequential_layer_call_and_return_conditional_losses_1639608§8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЭЮЫЬІ®AҐ>
7Ґ4
*К'
input_11€€€€€€€€€  
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ р
G__inference_sequential_layer_call_and_return_conditional_losses_1639723§8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЮЫЭЬІ®AҐ>
7Ґ4
*К'
input_11€€€€€€€€€  
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ о
G__inference_sequential_layer_call_and_return_conditional_losses_1640634Ґ8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЭЮЫЬІ®?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ о
G__inference_sequential_layer_call_and_return_conditional_losses_1640803Ґ8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЮЫЭЬІ®?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ »
,__inference_sequential_layer_call_fn_1639932Ч8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЭЮЫЬІ®AҐ>
7Ґ4
*К'
input_11€€€€€€€€€  
p

 
™ "К€€€€€€€€€
»
,__inference_sequential_layer_call_fn_1640140Ч8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЮЫЭЬІ®AҐ>
7Ґ4
*К'
input_11€€€€€€€€€  
p 

 
™ "К€€€€€€€€€
∆
,__inference_sequential_layer_call_fn_1640896Х8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЭЮЫЬІ®?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p

 
™ "К€€€€€€€€€
∆
,__inference_sequential_layer_call_fn_1640989Х8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЮЫЭЬІ®?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€  
p 

 
™ "К€€€€€€€€€
ё
%__inference_signature_wrapper_1640343і8%&'(-.4567DEKLMNSTZ[\]jkqrstyzАБВГФХЮЫЭЬІ®EҐB
Ґ 
;™8
6
input_11*К'
input_11€€€€€€€€€  "1™.
,
dense_1!К
dense_1€€€€€€€€€
