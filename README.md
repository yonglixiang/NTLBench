# NTLBench


![](figs/abstract.png)


We provide a systematically review for non-transferable learning (NTL) and introduce an unified framework for benchmarking NTL (dubbed as **NTLBench**). This figure shows the comparison of 5 methods (<font color="#7884AC">NTL</font>, <font color="#7884AC">CUTI-domain</font>, <font color="#7884AC">H-NTL</font>, <font color="#7884AC">SOPHON</font>, <font color="#7884AC">CUPI-domain</font>) on CIFAR \& STL with VGG-13, where we evaluate non-transferability performance and post-training robustness against 5 <font color="#E4785F">source domain fine-tuning (SourceFT)</font> attacks, 4 <font color="#8151BA">target domain fine-tuning (TargetFT)</font> attacks, and 6 <font color="#59A4B7">source-free domain adaptation (SFDA)</font> attacks (higher value means better performance/robustness). 

## Survey

![](figs/review.png)


## Benchmark

We include following methods:
- [[Paper](https://arxiv.org/pdf/2106.06916)][[Code](https://github.com/conditionWang/NTL)] `NTL` (ICLR 2022) 
- [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Model_Barrier_A_Compact_Un-Transferable_Isolation_Domain_for_Model_Intellectual_CVPR_2023_paper.pdf)][[Code](https://github.com/LyWang12/CUTI-Domain)] `CUTI-domain` (CVPR 2023) 
- [[Paper](https://openreview.net/pdf?id=FYKVPOHCpE)][[Code](https://github.com/tmllab/NTLBench)] `HNTL` (ICLR 2024) 
- [[Paper](https://arxiv.org/pdf/2404.12699)][[Code](https://github.com/ChiangE/Sophon)] `SOPHON` (IEEE S&P 2024)
- [[Paper](https://arxiv.org/pdf/2408.13161)][[Code](https://github.com/LyWang12/CUPI-Domain)] `CUPI-domain` (T-PAMI 2024)
- [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_Your_Transferability_Barrier_is_Fragile_Free-Lunch_for_Transferring_the_Non-Transferable_CVPR_2024_paper.pdf)][[Code](https://github.com/tmllab/2024_CVPR_TransNTL)] `TransNTL` (CVPR 2024) 

<!-- We propose the ﬁrst NTL benchmark (NTLBench), which contains a standard and uniﬁed training and evaluation process. NTLBench supports 5 SOTA NTL methods, 9 datasets (more than 116 domain pairs), 5 network architectures families, and 15 post-training attacks from 3 attack settings, providing more than 40,000 experimental conﬁgurations. -->

```
NTLBench will be released soon.
```