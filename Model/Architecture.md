## Architecture
### Frmae Stacking + Attention + MOE

##### finicial trading is garded as POMDP, thus we implement frame stacking as POMDP.

## DRL modeling isn't sufficient
## we need frame stacking

[frame1, frame1, frame1, frame1, frame1]

[frame1, frame1, frame1, frame1, frame2]

[frame1, frame2, frame3, frame4, frame5]


--
### NTP + GRPO

#### Probabalistic Discrecte Action Space:
#### the last network output is routed to three networks, eaech produces a probalisitc actino space

### [0.0,0.1,0.2,0.2,0.5]
### [0.0,2.7,0.2,0.9,0.5]
### [0.3,0.1,0.2,0.6,0.5]
---

learning over time: after first period of supervised learning, self imitating the data. Using GRPO, give scores for each group output and then learn all the data.

---

超级信息建模交易算法：GRPO+NTP+MOE+MLA+Cross Attention

MOE：穿越熊市、牛市、市场变化
跨越熊市和牛市周期、各种市场变化、整合各行各业Stock规律
从MEO开始考虑问题的

Dense MLP有非常好的泛化能力、不像MOE会存在信息孤岛。在大语言模型这种需要涌现能力的场景，比如说代码和数学的模式泛化和启发到诗歌和历史，MOE的隔离机制会成为信息交互的壁垒。

但是如果要做一个通用于所有企业的股票、不同的周期、不同的交易频率的AI体系，信息隔离是有意义的。（海量的模式识别专家相比超级函数拟合大概也更符合金融市场的直觉。请一个万能通才还是一群跨领域专家？）


代码：选择专家并发送、再然后就是通信了 DeepEP通信、GEMM计算。金融上一个路由网络重要。LoadBalance是到
点击

回到大模型：一个Shared Expert

但是在训练的时候会有赢家通吃的问题、退化成普通的Dense。如果用Axuiliary Loss强行分流的话会带来更多的训练麻烦，


如果是一个通用于
企业的信息（行业、特点）、周期（牛市、熊市）、频次（低频、中频、高频）、

有海量的模式组合

但是会有赢家通吃的问题、退化成普通的Dense。需要加入Axuliaary Loss。

GRPO：Beta中的Alpha Mining

金融市场很难有像围棋那样清晰的Game Tree、所以像是MCTS很难使用。

然而价值网络（就是评估函数）变得更加



结局：初创公司获得最完整的估值建模、最准确的资金流向、实现金融市场的有效性


MLA：Frame Stacking的维度爆炸

在信息量最全面的前提下选因子？开头的因子输入需要可解释性？

LangGraph可以形成另类数据因子体系？

如何量化另类数据也是一个问题、关乎Search的Query和Key？

DQL（Deep Q Learning）不会直接处理噪声巨大的市场数据，而是用Multi Factor作为DQL的状态输入。

然而，金融市场是明显的POMDP。因此，可以借鉴DeepMind的处理方法：Frame Stacking。



因子（Factor）质量越好、包含的信息量越全面，DQL对环境的理解就越准确。



金融DQL需要多因子模型，用Factor（因子）

NTP：跨周期交易的通用算法


Cross Attention：传统数据和另类数据的加权整合（Modality）

传统数据的可解释性比较好、可以从传统数据的可解释性规律出发，请求大模型搜索和发掘合适的另类数据信息补全。

Agentic System或许可以在处理和量化另类数据的同时、完成信息补全有完整逻辑链条的可解释摘要。

对于每一支不同的股票、需要补全的信息和逻辑或许有所不同。可以用金融理论分析的方法将所有另类数据打包和压缩到一套几组特定的维度，也可以针对每一支股票专门挖掘？怎么输入股票的特征信息？

这样的话，