?	À%W?h@À%W?h@!À%W?h@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsÀ%W?h@1????9???I$???@r0:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"B91.7 % of the total step time sampled is spent on 'Kernel Launch'.*noI?i?}W?V@Q??,D? @Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	????9???????9???!????9???*      ??!       2      ??!       :	$???@$???@!$???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?i?}W?V@y??,D? @?"u
Itraining/Adam/gradients/gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter???????!???????0"*
dense/MatMulMatMuly?}??!??Īl??0"f
3loss/dense_1_loss/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsvܯO???!fDNi/R??"-
conv2d/Relu_FusedConv2D???ξ???!Gg?????"_
8training/Adam/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdam4
??R???!????y???"X
<training/Adam/gradients/gradients/dense/MatMul_grad/MatMul_1MatMul.b6?IY??!0??"?<??"G
,dropout/cond/then/_0/OptionalFromValue_6/_84_Send	???7;??!??
???"G
,dropout/cond/then/_0/OptionalFromValue_1/_78_Send?K?0l???!?b??P2??"?
?training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_48/gradients/dropout/Mul_1_grad/Shape_1/OptionalGetValue/_122_SendUs?yך?!??K?????"?
?training/Adam/gradients/gradients/dropout/cond_grad/StatelessIf/then/_48/gradients/OptionalFromValue_1_grad/OptionalGetValue/_124_Send9<?<`ɚ?!?-?X???Q      Y@Y?&?l??<@aN6?d??Q@"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
highB91.7 % of the total step time sampled is spent on 'Kernel Launch'.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Pascal)(: B 