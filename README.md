# Dual-CoCoOp
고려대학교 COSE-461 25-1 자연어처리 Final Project의 논문 코드입니다.

--------
# Paper

## Title
Dual-CoCoOp : Dually-informed Conditional Context Optimization for Vision-Language Models

## Abstract
Prompt design critically influences the performance of CLIP and its derivatives, which motivates a surge of prompt learning methods. CoOp (Context Optimization) replaces the CLIP’s fixed textual template with learnable context tokens, yielding strong adaptation on limited data, but still suffering from domain specificity and poor generalization to unseen classes. Conditional CoOp (CoCoOp) alleviates this issue by generating image-dependent context tokens through a meta learning paradigm, yet relies exclusively on visual cues. This paper introduces Dual-CoCoOp (Dual Conditional Context Optimization), a prompt learning framework that jointly leverages vision-based and linguistic conditions via a gated fusion mechanism. For each class label, rich natural language descriptions are first synthesized with captioning models and subsequently distilled into compact linguistic tokens; these tokens are fused with image-conditioned tokens to form the final prompt. Extensive experiments on domain-specific benchmarks such as fine-grained aircraft recognition and remote sensing classification, demonstrate that Dual- CoCoOp consistently achieves higher accuracy than CoOp and CoCoOp. The results highlight that complementary visual and linguistic conditioning makes robust prompt optimization for vision language models. Code is available at https://github.com/lpaiu-cs/Dual-CoCoOp.
