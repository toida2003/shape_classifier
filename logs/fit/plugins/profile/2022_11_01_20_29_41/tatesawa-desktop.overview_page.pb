?	P?D??@P?D??@!P?D??@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsP?D??@1='?o|???Ih?
?O@r0:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"B86.8 % of the total step time sampled is spent on 'Kernel Launch'.*noI??"t?U@Q2??^?*@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	='?o|???='?o|???!='?o|???*      ??!       2      ??!       :	h?
?O@h?
?O@!h?
?O@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??"t?U@y2??^?*@?"u
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter?B??xۧ?!?B??xۧ?0"w
Ktraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter???L???!?.y?u??0"u
Jtraining/Adam/gradients/gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput?vűԥ?!%mi?/??0"f
3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits?o????!&e$e?W??"/
conv2d_1/Relu_FusedConv2DY*f%?O??!??}.?+??"-
conv2d/Relu_FusedConv2DZ??m??!??!l???"i
Htraining/Adam/gradients/gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGrad?v????!Y.X'???"X
:training/Adam/gradients/gradients/dense/MatMul_grad/MatMulMatMul?:G	?P??!	????0"*
dense/MatMulMatMulIH?'????!?6i*?Y??0"?
?training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_81/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue/_104_Send?7?:*h??!LH@|M??Q      Y@Y??????=@a????Q@"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
highB86.8 % of the total step time sampled is spent on 'Kernel Launch'.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 