?	uXᖏ?@uXᖏ?@!uXᖏ?@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsuXᖏ?@1H?Ȱ?7??I?5????@r0:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"B86.0 % of the total step time sampled is spent on 'Kernel Launch'.*noI?j??}U@Q?w?Tb,@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	H?Ȱ?7??H?Ȱ?7??!H?Ȱ?7??*      ??!       2      ??!       :	?5????@?5????@!?5????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?j??}U@y?w?Tb,@?"_
8training/Adam/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdam???Id???!???Id???"u
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter&4???Q??!f+?Y&/??0"*
dense/MatMulMatMul?'~ݡ??!ki?????0"f
3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits뤧?%~??!S?	????"X
:training/Adam/gradients/gradients/dense/MatMul_grad/MatMulMatMul@??[ʦ?!;??????0"-
conv2d/Relu_FusedConv2Dc?{_????!gW???3??"i
Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradz󕫳h??!??r?{??"X
<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1MatMulmd??G??!?L?P k??"g
Ldropout/cond/then/_0/dropout/Mul-0-0-TransposeNCHWToNHWC-LayoutOptimizer/_76_Send{??8?g??!.)L?x???"?
?training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_48/gradients/OptionalFromValue_1_grad/OptionalGetValue/_124_Send_??ǫ??!Ԅ˰????Q      Y@Y?&?l??<@aN6?d??Q@"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
highB86.0 % of the total step time sampled is spent on 'Kernel Launch'.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 