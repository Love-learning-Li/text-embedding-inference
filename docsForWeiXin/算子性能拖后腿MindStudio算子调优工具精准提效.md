![cover_image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/sYFP2LY06HFicT3oJClwyt4edDkEdch9PctSpce2ziaayicibaEJqTB8S3Kub9K2Ros5Goz3CSbiaz5AdSb2bR1reYw/0?wx_fmt=jpeg)

#  算子性能拖后腿？MindStudio算子调优工具精准提效

[ 昇腾AI开发者 ](javascript:void\(0\);)

_2025年09月04日 21:31_ __ _ _ _ _ _ 广东  _

以下文章来源于MindStudio  ，作者ZJB

![](http://wx.qlogo.cn/mmhead/Q3auHgzwzM5F6vVDy10wELxv717icwqcQwawHia2bjtxBnRqPlgnGfEw/0)
**MindStudio** .

华为昇腾MindStudio

![图片](https://mmbiz.qpic.cn/mmbiz_gif/OCxOD3xB4DUGuCjtTdgbpG3RkpVZeGiczq8KYJkllGjVmdBZRTibsJVNZApdiaibaf0wdibd0Aiagf6oI81k2gicib9ymA/640?wx_fmt=gif&from=appmsg#imgIndex=0)
** 0  ** ** 1  ** ** 背景与挑战  **

  

大模型背景下，各类模型不断涌现，AI应用场景的不断拓展，对NPU性能的要求愈发强烈，尤其是特定shape下，多数客户倾向自主开发高性能算子，过程中经常遇到如下难题：  

  
  

  

  * ** 算子性能成为模型效率瓶颈：  ** 随着深度学习模型复杂度提升，算子作为基础计算单元，其性能直接影响整个模型的推理、训练效率； 

  * ** 并行计算场景的复杂性：  ** NPU 支持多核心、多线程并行计算，算子在并行调度、数据分片等环节易出现性能损耗，性能瓶颈影响因素复杂； 

  * ** 优化策略选择困难：  ** 面对多种优化策略，缺少科学指导，依靠经验选择时间成本高、难以规模化复制，严重影响算子开发效率。 

  

算子性能受计算、存储、并行等多因素影响，传统工具难精准定位瓶颈，为此，昇腾 MindStudio 算子工具链提供了 msProf op
算子性能调优工具，能深度解析性能数据，助开发者高效突破瓶颈。

  

** 02  ** ** MindStudio  ** ** 算子调优工具技术解读  **

  

MindStudio算子开发工具（MindStudio OpDev Tools）是面向Ascend C编程语言的系列工具，所推出的msProf
op工具用于采集和分析运行在昇腾AI处理器上算子的关键性能指标，用户可根据输出的性能数据，快速定位算子的软、硬件性能瓶颈，提升算子性能的分析效率。

  

msProf op工具包含上板调优和仿真调优两种使用方式,
协助用户定位算子内存、算子代码以及算子指令的异常，实现全方位的算子调优。上板调优包含内存热力图、cache热力图、Roofline通算流水图等，仿真调优包含指令流水图、算子代码热点图等，关键特性介绍如下：

  

  

** 计算内存热力图：  ** 以资源维度展示算子基础信息、计算负载分析和内存负载分析的数据，协助开发者以全局视角识别资源瓶颈。

** Cache热力图：  ** 记录并可视化呈现Cache热力图，该热力图可显示对应指令信息，以便用户优化L2Cache命中率，从而优化算子程序

** Roofline瓶颈分析图：  ** 构建出处理器的性能模型，然后利用该性能模型快速评估出算子的理论性能极限，协助开发者快速识别瓶颈类型。

** 指令流水图：  ** 指令维度展示时序关系，并关联调用栈快速定位瓶颈位置。

** 算子代码热点图：  **
支持查看算子源码与指令集的映射关系、耗时情况等功能，可协助开发者识别热点代码分布，并分析热点函数优化的可行性。通算流水图：MC2算子（Matrix
Computation & Communication）通算运行情况、指令耗时等信息，协助开发者识别通算瓶颈。

  

** 0  ** ** 3  ** ** 经典案例与操作指南  **

  

在上一期算子精度调试案例中，我们已经介绍了如何通过异常检测与算子调试快速定位CrossEntropy算子的精度问题，本期，我们继续介绍如何利用msProf
op工具使用，进一步实现CrossEntropy算子性能调优。

  

  

** 1.环境准备  **

  

  

环境已安装CANN和MindStudio Insight，依据指导完成配置，确保算子可正常编译执行。

参考资料：算子调优工具：https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/optool/atlasopdev_16_0082.html

注：MindStudio
Insight是面向昇腾AI开发者的可视化调优工具，支持开发者在训练、推理以及算子开发场景快速完成性能瓶颈分析与优化。详细资料参见产品文档：
https://www.hiascend.com/document/detail/zh/mindstudio/81RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html。  
  

  

** 2.算子性能优化流程  **

  

  

使用算子调优工具，优化算子性能，主要流程如下：

建议先基于msProf op 上板获取性能结果，根据落盘数据提供的端到端耗时，AIV、AIC time等性能数据，分析当前性能是否达标；

对于不达标的性能算子，可基于msProf op simulator提供的仿真流水图功能，进一步分析性能瓶颈类型，识别优化点，并对代码进行修改

再次进入上板测试调优阶段；反复迭代，直至性能达预期。

  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9POuYoIXEYicmplHHcRib0KYcfesb0KC4S54Y322aV49gia0YbSODOFwoSg/640?wx_fmt=png&from=appmsg)

图2. 算子工具调优流程

  

  

** 3.算子案例介绍  **

  

  

CrossEntropy为一个交叉熵损失函数，[128, 1024]
shape下的性能数据如下表所示，算子总体时间约45.42us，对于一个VECTOR算子而言，该shape下性能相对偏差，本文将通过msProf op
算子性能调优工具识别性能瓶颈并进行优化。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9P1D3ibQyRr8CxoicelvTNvKdWjHoib2rjtHBTXOpibZurXNiaupC7icS8WpXA/640?wx_fmt=png&from=appmsg)

  

  

** 4.算子上板采集  **

  

  

使用msProf op 获取上板性能数据，参考命令：

  

  * 

    
    
    msprof op --aic-metrics=Occupancy,Default –application=<算子执行程序>

  

并将visualize_data.bin导入MindStudio Insight 查看上板性能。

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PmVoI4ejKplOCPyNPWQbHW19b3M51iaMicsvZEJoHZYuvE6MBudOYiaicGg/640?wx_fmt=png&from=appmsg)

图3 msProf op 性能采集结果  
  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PwlEHhZmCbBj2swH2Njb3ESsDYnCia4pia9vicYvVYhxekPvEEJxlSSHog/640?wx_fmt=png&from=appmsg)
![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PNibzYfyXP7Ktl8GfW5e811d4tHLYfRzDn8LcO58b8mG9pecf6Lgv0VQ/640?wx_fmt=png&from=appmsg)

图4. MindStudio Insight内存热力图部分  
  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PJpHWibewj097YgbP0v4HTSu9TjmodaeQeDEib53XHUEHqPiaZZiaSuSt4g/640?wx_fmt=png&from=appmsg)

图5. 计算和搬运单元数据  
  

依据上板内存热力图和落盘数据，算子为VECTOR算子，核间负载数据表明，该算子使用了40个vector块，计算分布均匀，均为绿色，可以排除负载不均问题。但通过内存负载分析，该算子scalar活跃率达87.86%，占比较高，表明kernel上存在大量scalar运算。图5“计算和搬运单元数据
”显示，算子scalar标量运算耗时较多，为当前性能瓶颈，可针对性优化。

  
  

** 5.算子仿真采集  **

  

  
在上板采集阶段，已经识别到scalar占比过高为主要性能瓶颈。那么，如何定位到scalar操作所对应的具体代码行，并进一步排查是否存在其他性能可优化点呢？可使用msprof
op simulator（算子仿真功能）获取仿真流水图，进一步定位性能代码行。参考命令：

  

  * 

    
    
    msprof op simulator --soc-version=Ascendxxyy  --application=<算子执行程序>

  

Ascendxxxyy需替换为实际使用的处理器类型。采集结果如下图所示：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PCPcicQ7IvOcW7v6ChUSxMoegryayVtYfXogibHjQkF0A0iaHBjE4g1Gibg/640?wx_fmt=png&from=appmsg)

图6. 算子仿真落盘数据

  

将工具落盘数据visualize_data.bin导入MindStudio Insight工具，查看流水图信息， 可视化结果如下图所示：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PdYibXKf4pzvodlHCl34Bt3qAtcG1ohFuSVibH2T6miaNuVK01A316nmBw/640?wx_fmt=png&from=appmsg)

图7. 仿真流水图  
  

通过仿真流水图，可以确认当前算子scalar计算较多，且发生在代码的109行。

  

  

** 6.算子性能优化-scalar优化  **

  
  
![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9Pyj3c6VoQG9twgfz6MU797r3UGZY2qbIxmky19B6tge122yvRXbMzBg/640?wx_fmt=png&from=appmsg)

图8. 代码热点图

  

通过代码热点图分析第109行代码逻辑，识别到当前算子性能瓶颈主要来源于将num_class进行OneHot，基于SetValue接口设置LocalTensor中的某个值，导致kernel上存在大量for循环和判断操作，即scalar运算。优化方案：使用Ascend
C 数据填充API
Duplicate，将一个变量或立即数复制多次并填充到向量中，并将最后一次的scalar设置1即可完成方案替换，进而减少scalar操作，代码修改如下：  
  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9P7bXFkBaCcIKprgic7LPZgUdicO2am1b3EMWsBgliawpB8GCD2icbkgBx0Q/640?wx_fmt=png&from=appmsg)

图9. Scalar代码优化对比图

  

优化后的流水图和上板数据如下：

  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PwxeoevpTlg8icWnUgrfKrCFtugibsfzJNhIXFbaRRRC1mwTEJfcVUzDA/640?wx_fmt=png&from=appmsg)

图10. Scalar优化后指令流水图

  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PbBgHIrpnpD9APoxJljJUftz0t4Kic4Pq3DmsRA4msN9s9dvutngA1Sw/640?wx_fmt=png&from=appmsg)

图11. Scalar优化后上板数据

  

通过指令流水图可以清晰看到，优化后的算子流水图scalar明显降低，各个pipe流水并行度较大提升。经过上板数据测试，scalar平均优化性能：16.71->4.04us；aiv_time平均优化性能20.95->10.68us，性能提升96.16%。

  

  

** 7.算子性能优化-代码逻辑优化  **

  

  

进一步分析scalar优化后流水图，发现两轮计算流水之间存在较大空隙（图10“Scalar优化后指令流水图”红框部分）。检查流水图代码调用栈，发现此处存在两个连续DataCopy操作，搬入阶段logitMovBytes存在冗余计算行为，可以将两次DataCopy合并为一次，节约搬运时间，修改方案如下：

  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9P6OrLtSFBq1IdIRXq1fWnDh3qMczUBIXPLrFL4GibZVNkIuicKk2EHU6Q/640?wx_fmt=png&from=appmsg)

图12. 重复计算代码优化对比图

  

![](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFicT3oJClwyt4edDkEdch9PAVF0M3OYGPiaDKhzIo0Mia2vRKPc31vsKu01MsrrB1qTbjUTBOuUwBJw/640?wx_fmt=png&from=appmsg)

图13. 重复计算优化后上板数据  
  

优化后，再次采集算子上板性能数据，得到scalar平均优化性能：
4.04->3.61us，aiv_time平均优化性能10.68->9.92us，性能提升7.6%。

  

** 0  ** ** 4  ** ** 总结  **

  

msProf op 是昇腾算子性能调优核心工具，支持上板与仿真两种调优模式，适配 Kernel 直调、AscendCL 单算子及 PyTorch
框架算子调用等场景，可采集计算单元利用率、内存、Cache 等关键指标，通过 MindStudio Insight 可视化呈现内存热力图、Roofline
图，助力开发者快速定位瓶颈，能有效解决算子开发中性能分析难问题，降低调优门槛，更多功能可参考算子开发工具详情资料介绍。

https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/optool/atlasopdev_16_0082.html

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFxyEfOW2Ohvgz6vEfHhPia4GibrDuX62NicM529lor4NP3OXaEenDQacPd2VXyhl1mfXibwPDsYZsAjg/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&randomid=rq20by2b&tp=webp#imgIndex=3)

预览时标签不可点

微信扫一扫  
关注该公众号



微信扫一扫  
使用小程序

****



****



****



×  分析

__

![作者头像](http://mmbiz.qpic.cn/mmbiz_png/sYFP2LY06HHfNmat26LtF17QUpsIPTThAhrBybenUoMdqXO0548R0rnxur54NibIx9bUktkolch4esv6gVgU7PA/0?wx_fmt=png)

微信扫一扫可打开此内容，  
使用完整服务

：  ，  ，  ，  ，  ，  ，  ，  ，  ，  ，  ，  ，  。  视频  小程序  赞  ，轻点两下取消赞  在看  ，轻点两下取消在看
分享  留言  收藏  听过

