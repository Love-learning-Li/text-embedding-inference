![cover_image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/sYFP2LY06HEZJQ9jxJfyNLSq2RsIVQZaSmEbBSzeISzsEpSR40VehThnyQSyxHRN7QagytmW8EqYFCpRhQ7aNQ/0?wx_fmt=jpeg)

#  SGLang全面支持昇腾，使能大EP高性能推理

[ 昇腾AI开发者 ](javascript:void\(0\);)

_2025年08月13日 20:01_ __ _ _ _ _ _ 广东  _

![](https://mmbiz.qpic.cn/mmbiz_gif/OCxOD3xB4DUGuCjtTdgbpG3RkpVZeGiczq8KYJkllGjVmdBZRTibsJVNZApdiaibaf0wdibd0Aiagf6oI81k2gicib9ymA/640?wx_fmt=gif&from=appmsg)
![](https://mmbiz.qpic.cn/mmbiz_png/7QRTvkK2qC63c02mKcsfAaJ8sNcicTvg22UkHHibvKiasFS9FS6E4FeV0Dibe7as7h4tm8p7EfNfI06adlGbL2icYjw/640?wx_fmt=png)

2025年8月，经过SGLang社区与昇腾的共同努力，将SGLang的灵活编程框架与昇腾强大的异构算力深度融合，使能SGLang在昇腾上无缝运行大模型推理，并正式面向用户推出基于SGLang的大EP集群推理解决方案。当前用户可获取最新release版本的SGLang以体验低延迟、高吞吐的推理系统。

  

  

#

  

** 多种类大模型支持，满足多样化需求  **

通过增强SGLang能力并集成昇腾异构加速推理能力，当前已可开箱支持多种稠密Dense、稀疏MoE大语言模型及多模态模型，如Qwen系列、LLaMA系列、DeepSeek系列等。用户可以在昇腾运行各类大模型推理，并基于SGLang增量开发，满足不同场景的应用需求。

  

  

** SGLang核心加速特性支持  **

经过SGLang社区与昇腾的协同，当前昇腾已支持部分核心加速特性，如PD Disaggregation、Overlap Scheduler、Tensor
Parallel、DP Attention、大EP推理，即将支持加速特性如EPLB、Speculative
Decoding及NPUGraph。社区最新加速能力平滑迁移至昇腾，面向用户提供高性能推理最优实践。

  

  

** 生态亲和加速的大EP推理框架  **

昇腾基于SGLang推理框架正式推出生态亲和的大EP推理加速库，通过北向接口兼容DeepEP，目标无需更改调用方式即可使能昇腾大EP方案，全面利用昇腾的技术优势，增强专家并行Expert
Parallel的吞吐能力。

  

  

** 共建昇腾生态库，加速开源开放  **

昇腾与SGLang社区共同创立sgl-project/sgl-kernel-
npu项目，面向SGLang社区提供生态亲和、面向全场景的标准接口融合算子与加速库。

  

  * 欢迎访问SGLang kernel library for NPU，链接： 

https://github.com/sgl-project/sgl-kernel-npu

  

![](https://mmbiz.qpic.cn/mmbiz_png/OCxOD3xB4DWfr8JPeopWHooKngy0xu6TbDMWdGudBcDETU4yTVT2qxkQKGdl9mz66pibErSBbwFC4RkZsdpRdIw/640?wx_fmt=png&from=appmsg)

  

  

** 快速体验  **

  * 容器准备 ： 

https://docs.sglang.ai/platforms/ascend_npu.html#method-2-using-docker

  

![](https://mmbiz.qpic.cn/mmbiz_png/OCxOD3xB4DWfr8JPeopWHooKngy0xu6Tk2JsW8F6WianDpwj83EtyF91gKiaBwpecBJNz2MMUdXN4zlKTclBkcWQ/640?wx_fmt=png&from=appmsg)

  

  * 快速体验DeepSeek ： 

https://docs.sglang.ai/platforms/ascend_npu.html#running-deepseek-v3

  

![](https://mmbiz.qpic.cn/mmbiz_png/OCxOD3xB4DWfr8JPeopWHooKngy0xu6TZaPJKfW3bMicM1aGKRqmya64WJywsmoIG8szozyNwf9EfibNCM9pckFQ/640?wx_fmt=png&from=appmsg)

  

  

** 致谢  **

SGLang全面支持昇腾大EP推理解决方案是昇腾生态发展的一个重要里程碑，这一成果的实现离不开社区的深度协作与卓越贡献。

  

我们在这里向以下给予支持与反馈的团队成员致以谢意：

  * SGLang核心团队：LianMin Zheng (https://github.com/merrymercy)、YiNeng Zhang (https://github.com/zhyncs)、JieXin Liang (https://github.com/Alcanderian)、ShangMing Cai (https://github.com/ShangmingCai) 感谢核心团队成员认真审核PR，给予优质的反馈意见并协助贡献高质量的代码 

  * KTransformers  团队：Mingxing Zhang，感谢分享基于昇腾平台的优化经验 

  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/sYFP2LY06HFxyEfOW2Ohvgz6vEfHhPia4GibrDuX62NicM529lor4NP3OXaEenDQacPd2VXyhl1mfXibwPDsYZsAjg/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&randomid=rq20by2b&tp=webp)

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

