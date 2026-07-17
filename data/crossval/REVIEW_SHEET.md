# Published-half review sheet — FINAL PUNCH LIST

Automation exhausted (abstract → HTML → PDF, extract-only). 32/50 complete.
Fill the ❌ fields below **in data/crossval/published_drafts.jsonl**, verify all 50, then promote.

## ⚠️ NEEDS YOUR FILL (18)

### pub-004  ❌ dataset, baselines
- **paper**: Navigating the State of Cognitive Flow: Context-Aware AI Interventions for Effective Reasoning Support
- **link**: https://arxiv.org/abs/2504.16021
- **venue claim**: Proceedings of the 2025 ACM CHI Workshop on Human-AI Interaction for Augmented Reasoning
- **venue evidence** [arxiv:journal_ref]: Proceedings of the 2025 ACM CHI Workshop on Human-AI Interaction for Augmented Reasoning
- motivation: In AI-augmented reasoning, interventions that disrupt the state of cognitive flow can hinder rather than enhance decision-making. This paper proposes a context-aware cognitive augmentation framework t
- hypothesis: By leveraging multimodal behavioral cues, AI can dynamically adjust cognitive support to maintain or restore flow.
- method_sketch: The approach introduces the concept of cognitive flow, an extension of flow theory in AI-augmented reasoning, where interventions are personalized, adaptive, and minimally intrusive. By shifting from 
- dataset: ❌ FILL FROM PAPER
- metrics: task performance and engagement metrics
- baselines: ❌ FILL FROM PAPER
- expected_outcome: The approach ensures that AI systems support deep engagement in complex decision-making and reasoning without disrupting cognitive immersion.

### pub-009  ❌ metrics
- **paper**: SVG: 3D Stereoscopic Video Generation via Denoising Frame Matrix
- **link**: https://arxiv.org/abs/2407.00367
- **venue claim**: ICLR 2025
- **venue evidence** [dblp]: {"venue": "ICLR", "year": "2025", "dblp_title": "SVG: 3D Stereoscopic Video Generation via Denoising Frame Matrix."}
- motivation: The generation of 3D stereoscopic video remains under-explored.
- hypothesis: A pose-free and training-free approach for generating 3D stereoscopic videos using an off-the-shelf monocular video generation model will be effective.
- method_sketch: The method warps a generated monocular video into camera views on stereoscopic baseline using estimated video depth, and employs a novel frame matrix video inpainting framework. The framework leverage
- dataset: videos from various generative models, including Sora, Lumiere, WALT, and Zeroscope
- metrics: ❌ FILL FROM PAPER
- baselines: previous methods
- expected_outcome: The method will generate consistent and semantically coherent stereoscopic videos without scene optimization or model fine-tuning.

### pub-011  ❌ metrics
- **paper**: ID-Booth: Identity-consistent Face Generation with Diffusion Models
- **link**: https://arxiv.org/abs/2504.07392
- **venue claim**: IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2025
- **venue evidence** [arxiv:journal_ref]: IEEE International Conference on Automatic Face and Gesture Recognition (FG), 2025
- motivation: State-of-the-art generative models typically rely on conditioning and fine-tuning of powerful pretrained diffusion models to facilitate the synthesis of realistic images of a desired identity. Yet, th
- hypothesis: A novel generative diffusion-based framework, called ID-Booth, can enable identity-consistent image generation while retaining the synthesis capabilities of pretrained diffusion models.
- method_sketch: ID-Booth consists of a denoising network responsible for data generation, a variational auto-encoder for mapping images to and from a lower-dimensional latent space and a text encoder that allows for 
- dataset: Tufts Face Database
- metrics: ❌ FILL FROM PAPER
- baselines: state-of-the-art latent diffusion model
- expected_outcome: Our method facilitates better intra-identity consistency and inter-identity separability than competing methods, while achieving higher image diversity.

### pub-013  ❌ metrics
- **paper**: Graph Tokenization for Bridging Graphs and Transformers
- **link**: https://arxiv.org/abs/2603.11099
- **venue claim**: per arXiv comment: Accepted as a poster at ICLR 2026. Code is available at https://github.com/BUPT-GAMMA/Graph-Tokeniza
- **venue evidence** [arxiv:comment]: Accepted as a poster at ICLR 2026. Code is available at https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers
- motivation: Extending large pretrained Transformers to graph-structured data remains a significant challenge.
- hypothesis: A graph tokenization framework that generates sequential representations of graphs by combining reversible graph serialization with Byte Pair Encoding (BPE) can enable Transformers to be directly appl
- method_sketch: We introduce a graph tokenization framework that combines reversible graph serialization, which preserves graph information, with BPE. The graph serialization process is guided by global statistics of
- dataset: 14 benchmark datasets
- metrics: ❌ FILL FROM PAPER
- baselines: graph neural networks, specialized graph transformers
- expected_outcome: The proposed approach achieves state-of-the-art results on the benchmark datasets and frequently outperforms both graph neural networks and specialized graph transformers.

### pub-015  ❌ dataset, baselines
- **paper**: Simulation of Language Evolution under Regulated Social Media Platforms: A Synergistic Approach of Large Language Models and Genetic Algorithms
- **link**: https://arxiv.org/abs/2502.19193
- **venue claim**: per arXiv comment: The manuscript has been accepted to IEEE Transactions on Computational Social Systems
- **venue evidence** [arxiv:comment]: The manuscript has been accepted to IEEE Transactions on Computational Social Systems
- motivation: Social media platforms frequently impose restrictive policies to moderate user content, prompting the emergence of creative evasion language strategies. This paper presents a multi-agent framework bas
- hypothesis: The framework will simulate the iterative evolution of language strategies under regulatory constraints.
- method_sketch: The framework uses participant agents as social media users who continuously evolve their language expression, while supervisory agents emulate platform-level regulation by assessing policy violations
- dataset: ❌ FILL FROM PAPER
- metrics: number of uninterrupted dialogue turns, accuracy of information transmission
- baselines: ❌ FILL FROM PAPER
- expected_outcome: As the number of dialogue rounds increases, both the number of uninterrupted dialogue turns and the accuracy of information transmission will improve significantly.

### pub-017  ❌ metrics
- **paper**: Hierarchical Pre-Training of Vision Encoders with Large Language Models
- **link**: https://arxiv.org/abs/2604.00086
- **venue claim**: per arXiv comment: 17 pages, 14 figures, accepted to Computer Vision and Pattern Recognition Conference (CVPR) Workshop
- **venue evidence** [arxiv:comment]: 17 pages, 14 figures, accepted to Computer Vision and Pattern Recognition Conference (CVPR) Workshops 2026. 5th MMFM Workshop: What is Next in Multimodal Foundation Models?
- motivation: Existing approaches often treat vision encoders and large language models (LLMs) as independent modules, limiting the integration of hierarchical visual features.
- hypothesis: HIVE (Hierarchical Pre-Training of Vision Encoders), a novel framework that enhances vision-language alignment by introducing hierarchical cross-attention between the vision encoder and LLM, will impr
- method_sketch: We propose HIVE, which enables structured feature fusion across multiple layers. To optimize this interaction, we introduce a three-stage training strategy that progressively aligns the vision encoder
- dataset: MME, GQA, OK-VQA, ScienceQA
- metrics: ❌ FILL FROM PAPER
- baselines: self-attention-based methods
- expected_outcome: HIVE achieves superior performance not only in image classification but also on various vision-language tasks.

### pub-018  ❌ dataset
- **paper**: Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents
- **link**: https://arxiv.org/abs/2507.22352
- **venue claim**: Proceedings of the 7th ACM Conference on Conversational User Interfaces (CUI '25), 2025, Article 49, 1-15
- **venue evidence** [arxiv:journal_ref]: Proceedings of the 7th ACM Conference on Conversational User Interfaces (CUI '25), 2025, Article 49, 1-15
- motivation: We investigated the challenges of mitigating response delays in free-form conversations with virtual agents powered by Large Language Models (LLMs) within Virtual Reality (VR). Our motivation is to op
- hypothesis: Latency above 4 seconds degrades quality of experience, while natural conversational fillers improve perceived response time, especially in high-delay conditions.
- method_sketch: We used conversational fillers, such as gestures and verbal cues, to bridge delays between user input and system responses and evaluate their effectiveness across various latency levels and interactio
- dataset: ❌ FILL FROM PAPER
- metrics: Response Time, Engagement, Good Impression, Discomfort
- baselines: None
- expected_outcome: Our findings provide insights for practitioners and researchers to optimize user engagement whenever conversational systems' responses are delayed.

### pub-019  ❌ dataset
- **paper**: Augmenting the action space with conventions to improve multi-agent cooperation in Hanabi
- **link**: https://arxiv.org/abs/2412.06333
- **venue claim**: Autonomous Agents and Multi-agent Systems, vol. 39, no. 28, (2025)
- **venue evidence** [arxiv:journal_ref]: Autonomous Agents and Multi-agent Systems, vol. 39, no. 28, (2025)
- motivation: The card game Hanabi is considered a strong medium for the testing and development of multi-agent reinforcement learning (MARL) algorithms, due to its cooperative nature, partial observability, limite
- hypothesis: Multi-agent problems containing partial observability, especially when limited communication is present, can benefit greatly from the use of implicit knowledge sharing.
- method_sketch: We propose a novel approach to augmenting an agent's action space using conventions, which act as a sequence of special cooperative actions that span over and include multiple time steps and multiple 
- dataset: ❌ FILL FROM PAPER
- metrics: exponential moving averages (with a weight value of 0.9995) and standard error of the mean
- baselines: Rainbow
- expected_outcome: This approach results in a significant improvement on the performance of existing techniques for self-play and cross-play for various number of cooperators within Hanabi.

### pub-026  ❌ dataset
- **paper**: Learning Fair Pareto-Optimal Policies in Multi-Objective Reinforcement Learning
- **link**: https://arxiv.org/abs/2606.18111
- **venue claim**: per arXiv comment: Accepted at the Reinforcement Learning Conference (RLC) 2025. 12 pages main + appendix, 8 figures, 4
- **venue evidence** [arxiv:comment]: Accepted at the Reinforcement Learning Conference (RLC) 2025. 12 pages main + appendix, 8 figures, 4 tables
- motivation: Fairness is an important aspect of decision-making in multi-objective reinforcement learning (MORL), where policies must ensure both optimality and equity across multiple, potentially conflicting obje
- hypothesis: The goal is to learn a set of Pareto-optimal policies that ensure fairness across all possible user preferences.
- method_sketch: We propose three novel algorithms, which include integrating GGF with multi-policy multi-objective Q-Learning (MOQL), state-augmented multi-policy MOQL for learning non-stationary policies, and its no
- dataset: ❌ FILL FROM PAPER
- metrics: GGF scores
- baselines: state-of-the-art MORL baselines
- expected_outcome: Our methods learn a set of fair policies that accommodate different user preferences.

### pub-027  ❌ metrics
- **paper**: Prompting Video-Language Foundation Models with Domain-specific Fine-grained Heuristics for Video Question Answering
- **link**: https://arxiv.org/abs/2410.09380
- **venue claim**: IEEE Transactions on Circuits and Systems for Video Technology, 2024
- **venue evidence** [arxiv:journal_ref]: IEEE Transactions on Circuits and Systems for Video Technology, 2024
- motivation: Despite advancements in multi-modal pre-trained models and video-language foundation models, these systems often struggle with domain-specific VideoQA due to their generalized pre-training objectives.
- hypothesis: Leveraging domain-specific entity-action heuristics can refine pre-trained video-language foundation models to enhance reasoning capabilities.
- method_sketch: We introduce HeurVidQA, a framework that treats these models as implicit knowledge engines, employing domain-specific entity-action prompters to direct the model's focus toward precise cues that enhan
- dataset: multiple VideoQA datasets
- metrics: ❌ FILL FROM PAPER
- baselines: existing models
- expected_outcome: Our method significantly outperforms existing models, underscoring the importance of integrating domain-specific knowledge into video-language models for more accurate and context-aware VideoQA.

### pub-031  ❌ metrics
- **paper**: Improving Graph Embeddings in Machine Learning Using Knowledge Completion with Validation in a Case Study on COVID-19 Spread
- **link**: https://arxiv.org/abs/2511.12071
- **venue claim**: 2025 IEEE International Conference on Knowledge Graphs (ICKG)
- **venue evidence** [arxiv:journal_ref]: 2025 IEEE International Conference on Knowledge Graphs (ICKG)
- motivation: Graph embeddings (GEs) are derived from explicit topology and features, they may miss crucial implicit knowledge hidden in seemingly sparse datasets, affecting graph structure and their representation
- hypothesis: Integrating a Knowledge Completion (KC) phase to uncover latent dataset semantics before embedding generation can improve graph embeddings in machine learning.
- method_sketch: We propose a GML pipeline that integrates a Knowledge Completion (KC) phase to uncover latent dataset semantics before embedding generation. Focusing on transitive relations, we model hidden connectio
- dataset: real-world COVID-19 contact network
- metrics: ❌ FILL FROM PAPER
- baselines: GraphSAGE, Node2Vec
- expected_outcome: Our GML pipeline significantly alters the embedding space geometry, demonstrating that its introduction is not just a simple enrichment but a transformative step that redefines graph representation qu

### pub-034  ❌ baselines
- **paper**: Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments
- **link**: https://arxiv.org/abs/2401.09862
- **venue claim**: per arXiv comment: Accepted in EvoApps at EvoStar 2024
- **venue evidence** [arxiv:comment]: Accepted in EvoApps at EvoStar 2024
- motivation: The importance of effective prompt engineering has come to the fore as the use of large language models (LLMs) continues to grow, as it has a direct impact on model performance and the extraction of r
- hypothesis: Evolutionary algorithms (EAs) can be used to optimize prompts for LLMs.
- method_sketch: We propose a evolutionary multi-objective (EMO) approach specifically tailored for prompt optimization called EMO-Prompts, using sentiment analysis as a case study.
- dataset: emotion dataset
- metrics: hypervolume
- baselines: ❌ FILL FROM PAPER
- expected_outcome: EMO-Prompts effectively generates prompts capable of guiding the LLM to produce texts embodying two conflicting emotions simultaneously.

### pub-035  ❌ metrics, baselines
- **paper**: Chitrakshara: A Large Multilingual Multimodal Dataset for Indian languages
- **link**: https://arxiv.org/abs/2603.23521
- **venue claim**: per arXiv comment: Accepted at "CVPR 2025: Workshop Vision Language Models For All"
- **venue evidence** [arxiv:comment]: Accepted at "CVPR 2025: Workshop Vision Language Models For All"
- motivation: Multimodal research has predominantly focused on single-image reasoning, with limited exploration of multi-image scenarios. Most Vision-Language Models (VLMs) are trained primarily on English datasets
- hypothesis: The Chitrakshara dataset series, covering 11 Indian languages, can help develop more culturally inclusive VLMs.
- method_sketch: We introduce the Chitrakshara dataset series, comprising Chitrakshara-IL with 193M images, 30B text tokens, and 50M multilingual documents, and Chitrakshara-Cap with 44M image-text pairs and 733M toke
- dataset: Chitrakshara-IL, Chitrakshara-Cap
- metrics: ❌ FILL FROM PAPER
- baselines: ❌ FILL FROM PAPER
- expected_outcome: The dataset's representativeness across Indic languages and its potential for developing more culturally inclusive VLMs will be assessed.

### pub-036  ❌ dataset
- **paper**: Beyond Words: Infusing Conversational Agents with Human-like Typing Behaviors
- **link**: https://arxiv.org/abs/2510.08912
- **venue claim**: CUI '24: Proceedings of the ACM Conversational User Interfaces 2024, July 8-10, 2024, Luxembourg, Luxembourg. ACM, New Y
- **venue evidence** [arxiv:journal_ref]: CUI '24: Proceedings of the ACM Conversational User Interfaces 2024, July 8-10, 2024, Luxembourg, Luxembourg. ACM, New York, NY, USA, 11 pages
- motivation: A notable distinction between human-like dialogues and AI models is that AI models predominantly generate responses rapidly, often producing extensive content without emulating the thoughtful process 
- hypothesis: An agent with human-like typing behaviors could potentially affect conversational engagement and its trustworthiness.
- method_sketch: We've constructed an interactive platform featuring user-adjustable parameters, allowing users to personalize the AI's communication style. Our user experiment involves interactions with three types o
- dataset: ❌ FILL FROM PAPER
- metrics: subjective metrics
- baselines: baseline agent, one simulating hesitation, and another integrating both hesitation and self-editing behaviors
- expected_outcome: The agent that incorporates both behaviors could be preferred, suggesting an improvement in perceived naturalness and trustworthiness.

### pub-037  ❌ metrics
- **paper**: AOAD-MAT: Transformer-based multi-agent deep reinforcement learning model considering agents' order of action decisions
- **link**: https://arxiv.org/abs/2510.13343
- **venue claim**: PRIMA 2025: Principles and Practice of Multi-Agent Systems, LNCS 16366, pp. 303-310 (2026)
- **venue evidence** [arxiv:journal_ref]: PRIMA 2025: Principles and Practice of Multi-Agent Systems, LNCS 16366, pp. 303-310 (2026)
- motivation: Multi-agent reinforcement learning focuses on training the behaviors of multiple learning agents that coexist in a shared environment, but existing models do not explicitly consider the importance of 
- hypothesis: A novel MAT model that considers the order in which agents make decisions will outperform existing models.
- method_sketch: We propose an Agent Order of Action Decisions-MAT (AOAD-MAT), a novel MAT model that considers the order in which agents make decisions. The proposed model explicitly incorporates the sequence of acti
- dataset: StarCraft Multi-Agent Challenge, Multi-Agent MuJoCo benchmarks
- metrics: ❌ FILL FROM PAPER
- baselines: MAT, ACE
- expected_outcome: The proposed AOAD-MAT model will outperform existing MAT and other baseline models.

### pub-038  ❌ metrics
- **paper**: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents
- **link**: https://arxiv.org/abs/2507.22925
- **venue claim**: EACL 2026
- **venue evidence** [dblp]: {"venue": "EACL", "year": "2026", "dblp_title": "H-MEM: Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents."}
- motivation: Long-term memory is one of the key factors influencing the reasoning capabilities of Large Language Model Agents (LLM Agents). Incorporating a memory mechanism that effectively integrates past interac
- hypothesis: We propose a Hierarchical Memory (H-MEM) architecture for LLM Agents that organizes and updates memory in a multi-level fashion based on the degree of semantic abstraction.
- method_sketch: Each memory vector is embedded with a positional index encoding pointing to its semantically related sub-memories in the next layer. During the reasoning phase, an index-based routing mechanism enable
- dataset: LoCoMo dataset
- metrics: ❌ FILL FROM PAPER
- baselines: five baseline methods
- expected_outcome: Our approach consistently outperforms five baseline methods, demonstrating its effectiveness in long-term dialogue scenarios.

### pub-043  ❌ metrics, baselines
- **paper**: Reviewing Clinical Knowledge in Medical Large Language Models: Training and Beyond
- **link**: https://arxiv.org/abs/2502.20988
- **venue claim**: Knowledge-Based Systems, 114215(2025)
- **venue evidence** [arxiv:journal_ref]: Knowledge-Based Systems, 114215(2025)
- motivation: The large-scale development of large language models (LLMs) in medical contexts necessitates that these models possess accurate medical knowledge and deliver traceable decision-making processes. Clini
- hypothesis: Various initiatives can embed clinical knowledge into training-based, KG-supported, and RAG-assisted LLMs.
- method_sketch: We review the various initiatives to embed clinical knowledge into LLMs. We gather reliable knowledge sources from the medical domain, evaluate implementations for integrating clinical knowledge throu
- dataset: databases, datasets
- metrics: ❌ FILL FROM PAPER
- baselines: ❌ FILL FROM PAPER
- expected_outcome: We will present evaluation systems applicable to relevant tasks and identify potential challenges facing this field.

### pub-044  ❌ metrics
- **paper**: Stabilizing Extreme Q-learning by Maclaurin Expansion
- **link**: https://arxiv.org/abs/2406.04896
- **venue claim**: Reinforcement Learning Journal, 2024, Volume 3, pages 1427-1440
- **venue evidence** [arxiv:journal_ref]: Reinforcement Learning Journal, 2024, Volume 3, pages 1427-1440
- motivation: In offline reinforcement learning, in-sample learning methods have been widely used to prevent performance degradation caused by evaluating out-of-distribution actions from the dataset. Extreme Q-lear
- hypothesis: Applying Maclaurin expansion to the loss function in XQL enhances stability against large errors.
- method_sketch: This approach involves adjusting the modeled value function between the value function under the behavior policy and the soft optimal value function, thus achieving a trade-off between stability and o
- dataset: DM Control, D4RL
- metrics: ❌ FILL FROM PAPER
- baselines: Extreme Q-learning (XQL)
- expected_outcome: Our method significantly stabilizes learning in online RL tasks from DM Control, where XQL was previously unstable. Additionally, it improves performance in several offline RL tasks from D4RL.

## ✅ Complete — verify only (32)

### pub-001  ✅
- **paper**: ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment
- **link**: https://arxiv.org/abs/2512.06196
- **venue claim**: per arXiv comment: Accepted to the AAAI 2026 LLAMAS Workshop (Large Language Model Agents for Multi-Agent Systems)
- **venue evidence** [arxiv:comment]: Accepted to the AAAI 2026 LLAMAS Workshop (Large Language Model Agents for Multi-Agent Systems)
- motivation: Maintaining alignment with stakeholder preferences is critical for agents based on large language models deployed to long-horizon tasks. Effective alignment requires reward models that are interpretab
- hypothesis: Rubric-based reward models offer a promising path toward interpretable, test-time adaptive alignment for complex, long-horizon AI systems.
- method_sketch: We introduce ARCANE, a framework that frames alignment as a multi-agent collaboration problem that dynamically represents stakeholder preferences as natural-language rubrics. We formulate rubric learn
- dataset: GDPVal benchmark
- metrics: compactness, legibility, correctness, conciseness
- baselines: Baselines.
- expected_outcome: The learned rubrics will produce compact, legible evaluations and enable configurable trade-offs without retraining.

### pub-002  ✅
- **paper**: Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement
- **link**: https://arxiv.org/abs/2512.18950
- **venue claim**: per arXiv comment: Accepted at The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 20
- **venue evidence** [arxiv:comment]: Accepted at The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2026). 21 pages including references, with 7 figures and 8 tables. Code is publicl
- motivation: The paper presents a framework that decouples reasoning from learning by maintaining a frozen large language model while performing all adaptation in an external hierarchical procedural memory.
- hypothesis: Structured external memory with Bayesian selection and contrastive refinement enables sample-efficient, interpretable, and continually improving agents without LLM parameter updates.
- method_sketch: MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes and 
- dataset: ALFWorld, WebShop, TravelPlanner, InterCodeSQL
- metrics: average performance, positive generalization
- baselines: all baselines, state-of-the-art LLM parameter-training baseline
- expected_outcome: The system constructs memory in 56 seconds, 2800 times faster than the state-of-the-art LLM parameter-training baseline, compressing 2851 trajectories into 187 procedures.

### pub-003  ✅
- **paper**: CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning
- **link**: https://arxiv.org/abs/2411.16313
- **venue claim**: per arXiv comment: Accepted to ICCV 2025. Codes and dataset are available at: https://github.com/duowuyms/OpenCATP-LLM
- **venue evidence** [arxiv:comment]: Accepted to ICCV 2025. Codes and dataset are available at: https://github.com/duowuyms/OpenCATP-LLM
- motivation: Prior studies overlook the tool execution costs, leading to the generation of expensive plans whose costs outweigh their benefits in terms of task performance.
- hypothesis: LLMs can be empowered for cost-aware tool planning.
- method_sketch: We design a tool planning language to enhance the LLM for creating multi-branch non-sequential plans. Moreover, we propose a cost-aware offline reinforcement learning algorithm to fine-tune the LLM to
- dataset: OpenCATP
- metrics: plan quality
- baselines: GPT-4, Llama2-7B
- expected_outcome: CATP-LLM outperforms baselines in terms of plan quality.

### pub-005  ✅
- **paper**: RAGPart & RAGMask: Retrieval-Stage Defenses Against Corpus Poisoning in Retrieval-Augmented Generation
- **link**: https://arxiv.org/abs/2512.24268
- **venue claim**: per arXiv comment: Published at AAAI 2026 Workshop on New Frontiers in Information Retrieval [Oral]
- **venue evidence** [arxiv:comment]: Published at AAAI 2026 Workshop on New Frontiers in Information Retrieval [Oral]
- motivation: Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to enhance large language models (LLMs) with external knowledge, reducing hallucinations and compensating for outdated informat
- hypothesis: We propose two complementary retrieval-stage defenses: RAGPart and RAGMask. Our defenses operate directly on the retriever, making them computationally lightweight and requiring no modification to the
- method_sketch: RAGPart leverages the inherent training dynamics of dense retrievers, exploiting document partitioning to mitigate the effect of poisoned points. In contrast, RAGMask identifies suspicious tokens base
- dataset: two benchmarks
- metrics: attack success rates
- baselines: Naive Combination Baseline
- expected_outcome: Our defenses consistently reduce attack success rates while preserving utility under benign conditions.

### pub-006  ✅
- **paper**: Investigating Retrieval-Augmented Generation in Quranic Studies: A Study of 13 Open-Source Large Language Models
- **link**: https://arxiv.org/abs/2503.16581
- **venue claim**: International Journal of Advanced Computer Science and Applications(IJACSA), 16(2), 2025
- **venue evidence** [arxiv:journal_ref]: International Journal of Advanced Computer Science and Applications(IJACSA), 16(2), 2025
- motivation: Accurate and contextually faithful responses are critical when applying large language models (LLMs) to sensitive and domain-specific tasks, such as answering queries related to quranic studies. Gener
- hypothesis: A Retrieval-Augmented Generation (RAG) is used to make up for the problems that come with using separate models.
- method_sketch: This research utilizes a descriptive dataset of Quranic surahs including the meanings, historical context, and qualities of the 114 surahs, allowing the model to gather relevant knowledge before respo
- dataset: descriptive dataset of Quranic surahs
- metrics: context relevance, answer faithfulness, answer relevance
- baselines: 13 open-source LLMs categorized into large (e.g., Llama3:70b, Gemma2:27b, QwQ:32b), medium (e.g., Gemma2:9b, Llama3:8b), and small (e.g., Llama3.2:3b, Phi3:3.8b)
- expected_outcome: The findings reveal that large models consistently outperform smaller models in capturing query semantics and producing accurate, contextually grounded responses.

### pub-007  ✅
- **paper**: DeepFusion: Accelerating MoE Training via Federated Knowledge Distillation from Heterogeneous Edge Devices
- **link**: https://arxiv.org/abs/2602.14301
- **venue claim**: Open MIND 2026
- **venue evidence** [openalex]: {"venue": "Open MIND", "year": "2026", "oa_title": "DeepFusion: Accelerating MoE Training via Federated Knowledge Distillation from Heterogeneous Edge Devices"}
- motivation: Mixture-of-Experts (MoE)-based large language models (LLMs) require vast and diverse training data, and traditional federated learning approaches are impractical for resource-constrained devices due t
- hypothesis: DeepFusion, a scalable federated MoE training framework, can enable the fusion of heterogeneous on-device LLM knowledge via federated knowledge distillation, yielding a knowledge-abundant global MoE m
- method_sketch: DeepFusion features each device to independently configure and train an on-device LLM tailored to its own needs and hardware limitations. It proposes a novel View-Aligned Attention (VAA) module that i
- dataset: medical and finance
- metrics: communication costs, token perplexity
- baselines: centralized MoE training, key federated MoE baselines
- expected_outcome: DeepFusion achieves performance close to centralized MoE training, reduces communication costs by up to 71%, and improves token perplexity by up to 5.28%.

### pub-008  ✅
- **paper**: Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning
- **link**: https://arxiv.org/abs/2602.12375
- **venue claim**: Reinforcement Learning Journal, vol. 6, 2025, pp. 1894-1915
- **venue evidence** [arxiv:journal_ref]: Reinforcement Learning Journal, vol. 6, 2025, pp. 1894-1915
- motivation: Optimistic value estimates provide one mechanism for directed exploration in reinforcement learning (RL), but this approach only increases the value bonus for an action retroactively, after seeing a h
- hypothesis: VBE uses the errors in the estimation of these RQFs to design value bonuses that provide first-visit optimism and deep exploration.
- method_sketch: We introduce an algorithm for exploration called Value Bonuses with Ensemble errors (VBE), that maintains an ensemble of random action-value functions (RQFs). The key idea is to design the rewards for
- dataset: several classic environments used to test exploration, Atari
- metrics: accumulated reward over learning
- baselines: Bootstrap DQN, RND, ACB
- expected_outcome: VBE outperforms Bootstrap DQN and two reward bonus approaches (RND and ACB) on several classic environments used to test exploration and can scale easily to more complex environments like Atari.

### pub-010  ✅
- **paper**: FBSDiff: Plug-and-Play Frequency Band Substitution of Diffusion Features for Highly Controllable Text-Driven Image Translation
- **link**: https://arxiv.org/abs/2408.00998
- **venue claim**: per arXiv comment: Accepted conference paper of ACM MM 2024
- **venue evidence** [arxiv:comment]: Accepted conference paper of ACM MM 2024
- motivation: Lacking controllability of large-scale text-to-image diffusion models restricts their practical applicability for real-life content creation.
- hypothesis: A plug-and-play frequency band substitution of diffusion features can adapt pre-trained large-scale text-to-image diffusion model to the image-to-image paradigm, realizing high-quality and versatile t
- method_sketch: The approach decomposes diverse guiding factors with different frequency bands of diffusion features in the DCT spectral space, and devises a novel frequency band substitution layer which realizes dyn
- dataset: LAION Aesthetics 6.5+
- metrics: Structure Similarity, Perceptual Similarity, Style Distance, CLIP Similarity, Aesthetic Score
- baselines: related methods
- expected_outcome: The approach will allow flexible control over both guiding factor and guiding intensity of the reference image simply by tuning the type and bandwidth of the substituted frequency band, and will outpe

### pub-012  ✅
- **paper**: Newton-Puiseux Analysis for Interpretability and Calibration of Complex-Valued Neural Networks
- **link**: https://arxiv.org/abs/2504.19176
- **venue claim**: Neural Networks, Volume 195, 2026, Article 108172
- **venue evidence** [arxiv:journal_ref]: Neural Networks, Volume 195, 2026, Article 108172
- motivation: Complex-valued neural networks (CVNNs) are particularly suitable for handling phase-sensitive signals, including electrocardiography (ECG), radar/sonar, and wireless in-phase/quadrature (I/Q) streams,
- hypothesis: A Newton-Puiseux framework that examines the local decision geometry of a trained CVNN can enhance Expected Calibration Error.
- method_sketch: The method involves fitting a small, kink-aware polynomial surrogate to the logit difference in the vicinity of uncertain inputs, and factorizing this surrogate using Newton-Puiseux expansions to deri
- dataset: MIT-BIH arrhythmia (ECG) dataset, RadioML 2016.10a (wireless modulation), controlled C^2 synthetic benchmark
- metrics: Expected Calibration Error
- baselines: uncalibrated softmax, standard post-hoc baselines
- expected_outcome: The method will enhance Expected Calibration Error in two case studies beyond a controlled C^2 synthetic benchmark.

### pub-014  ✅
- **paper**: Graph Neural Network Training Systems: A Performance Comparison of Full-Graph and Mini-Batch
- **link**: https://arxiv.org/abs/2406.00552
- **venue claim**: Proc. VLDB Endow. 2024
- **venue evidence** [dblp]: {"venue": "Proc. VLDB Endow.", "year": "2024", "dblp_title": "Graph Neural Network Training Systems: A Performance Comparison of Full-Graph and Mini-Batch."}
- motivation: Since two common methods for training GNNs require different training pipelines and systems optimizations, two separate classes of GNN training systems emerged, each tailored for one method. Works tha
- hypothesis: The mini-batch training systems consistently converge faster than the full-graph training ones across multiple datasets, GNN models, and system configurations.
- method_sketch: We provide a comprehensive empirical comparison of representative full-graph and mini-batch GNN training systems.
- dataset: multiple datasets
- metrics: time-to-accuracy
- baselines: other systems within the same category
- expected_outcome: mini-batch training techniques converge to similar to or often higher accuracy values than full-graph training ones

### pub-016  ✅
- **paper**: Evolutionary Pre-Prompt Optimization for Mathematical Reasoning
- **link**: https://arxiv.org/abs/2412.04291
- **venue claim**: per arXiv comment: Revised and extended version. To appear in ACM Transactions on Evolutionary Learning and Optimizatio
- **venue evidence** [arxiv:comment]: Revised and extended version. To appear in ACM Transactions on Evolutionary Learning and Optimization (TELO)
- motivation: This paper explores the optimization of example selection for designing effective CoT pre-prompts and shows that the choice of the optimization algorithm significantly enhances efficacy and feasibilit
- hypothesis: Evolutionary Pre-Prompt Optimization (EPPO) brings an improvement over the naive few-shot approach.
- method_sketch: The paper uses evolutionary computation for optimizing example selection for designing effective CoT pre-prompts.
- dataset: GSM8k, MathQA
- metrics: exact match scores
- baselines: naive few-shot approach
- expected_outcome: EPPO exceeding 10 absolute points in exact match scores on benchmark datasets.

### pub-020  ✅
- **paper**: Preference-Aware Memory Update for Long-Term LLM Agents
- **link**: https://arxiv.org/abs/2510.09720
- **venue claim**: Annual Meeting of the Association for Computational Linguistics 2025
- **venue evidence** [semantic_scholar]: {"venue": "Annual Meeting of the Association for Computational Linguistics", "year": "2025", "s2_title": "Preference-Aware Memory Update for Long-Term LLM Agents"}
- motivation: One of the key factors influencing the reasoning capabilities of LLM-based agents is their ability to leverage long-term memory. Integrating long-term memory mechanisms allows agents to make informed 
- hypothesis: By integrating sliding window averages (SW) with exponential moving averages (EMA), PAMU constructs a fused preference-aware representation that captures both short-term fluctuations and long-term use
- method_sketch: We propose a Preference-Aware Memory Update Mechanism (PAMU) that enables dynamic and personalized memory refinement. PAMU integrates sliding window averages (SW) with exponential moving averages (EMA
- dataset: LoCoMo
- metrics: F1 Score
- baselines: five baselines
- expected_outcome: Our mechanism can significantly improve the output quality of LLM in long-term conversations.

### pub-021  ✅
- **paper**: ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents
- **link**: https://arxiv.org/abs/2607.04686
- **venue claim**: per arXiv comment: 18 pages, 3 figures. Published at the Workshop on Agents in the Wild: Safety, Security, and Beyond (
- **venue evidence** [arxiv:comment]: 18 pages, 3 figures. Published at the Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD) and the Workshop on Failure Modes of Agentic AI (FAGEN) at ICML 2026
- motivation: Aggregate benchmark scores often hide where tool use fails in modern language model agents. A model that never calls a needed tool and a model that calls the tool but ignores the result can look simil
- hypothesis: Faithful tool use is not saturated and models with similar aggregate scores fail in different ways.
- method_sketch: The authors introduce ToolFailBench, a diagnostic benchmark for measuring tool-use failures across 1,000 tasks in finance, medicine, law, cybersecurity, and real estate. They label each trace with Too
- dataset: 1,000 tasks in finance, medicine, law, cybersecurity, and real estate
- metrics: Clean Tool-Use Rate, control-task accuracy
- baselines: 19 headline models
- expected_outcome: The best model reaches 86.33% Clean Tool-Use Rate, showing that faithful tool use is not saturated and models with similar aggregate scores fail in different ways.

### pub-022  ✅
- **paper**: CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization
- **link**: https://arxiv.org/abs/2604.14214
- **venue claim**: per arXiv comment: Accepted at ICLR 2026 Workshop on Logical Reasoning of Large Language Models
- **venue evidence** [arxiv:comment]: Accepted at ICLR 2026 Workshop on Logical Reasoning of Large Language Models
- motivation: Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation.
- hypothesis: Cost-Regularized Optimization of Prompts (CROP), an APO method that introduces regularization on response length by generating textual feedback in addition to standard accuracy feedback, will produce 
- method_sketch: We propose Cost-Regularized Optimization of Prompts (CROP), an APO method that introduces regularization on response length by generating textual feedback in addition to standard accuracy feedback. Th
- dataset: GSM8K, LogiQA, BIG-Bench Hard
- metrics: token consumption, accuracy
- baselines: standard automatic prompt optimization frameworks
- expected_outcome: CROP will achieve a significant reduction in token consumption while maintaining competitive accuracy.

### pub-023  ✅
- **paper**: Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval
- **link**: https://arxiv.org/abs/2504.05181
- **venue claim**: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '
- **venue evidence** [arxiv:journal_ref]: Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), pages 1327-1338, 2025
- motivation: Existing Generative information retrieval (GenIR) models suffer from token-level misalignment, where models trained to predict the next token often fail to capture document-level relevance effectively
- hypothesis: Direct document relevance optimization (DDRO) can align token-level docid generation with document-level relevance estimation through direct optimization via pairwise ranking, eliminating the need for
- method_sketch: We propose direct document relevance optimization (DDRO), which eliminates the need for explicit reward modeling and reinforcement learning by framing alignment as a direct optimization problem.
- dataset: MS MARCO document, Natural Questions
- metrics: MRR@10
- baselines: reinforcement learning-based methods
- expected_outcome: DDRO outperforms reinforcement learning-based methods, enhancing retrieval effectiveness with a simplified optimization approach.

### pub-024  ✅
- **paper**: PediatricsGPT: Large Language Models as Chinese Medical Assistants for Pediatric Applications
- **link**: https://arxiv.org/abs/2405.19266
- **venue claim**: per arXiv comment: Accepted by NeurIPS 2024. A Technical Report on a Chinese Medical Large Language Model
- **venue evidence** [arxiv:comment]: Accepted by NeurIPS 2024. A Technical Report on a Chinese Medical Large Language Model
- motivation: Developing intelligent pediatric consultation systems offers promising prospects for improving diagnostic efficiency, especially in China, where healthcare resources are scarce.
- hypothesis: PediatricsGPT, the first Chinese pediatric LLM assistant built on a systematic and robust training pipeline, will outperform previous Chinese medical LLMs.
- method_sketch: We build PedCorpus, a high-quality dataset of over 300,000 multi-task instructions from pediatric textbooks, guidelines, and knowledge graph resources. We propose PediatricsGPT, which includes a hybri
- dataset: PedCorpus
- metrics: GPT-4, doctor evaluations
- baselines: previous Chinese medical LLMs
- expected_outcome: PediatricsGPT consistently outperforms previous Chinese medical LLMs

### pub-025  ✅
- **paper**: DistillLens: Symmetric Knowledge Distillation Through Logit Lens
- **link**: https://arxiv.org/abs/2602.13567
- **venue claim**: Open MIND 2026
- **venue evidence** [openalex]: {"venue": "Open MIND", "year": "2026", "oa_title": "DistillLens: Symmetric Knowledge Distillation Through Logit Lens"}
- motivation: Standard Knowledge Distillation (KD) compresses Large Language Models (LLMs) by optimizing final outputs, yet it typically treats the teacher's intermediate layer's thought process as a black box. Whi
- hypothesis: By projecting intermediate hidden states into the vocabulary space via the Logit Lens, we enforce structural alignment using a symmetric divergence objective.
- method_sketch: We introduce DistillLens, a framework that symmetrically aligns the evolving thought processes of student and teacher models. Our analysis proves that this constraint imposes a dual-sided penalty, pre
- dataset: GPT-2, Llama
- metrics: instruction-following benchmarks
- baselines: standard KD, feature-transfer
- expected_outcome: DistillLens consistently outperforms standard KD and feature-transfer baselines

### pub-028  ✅
- **paper**: Reusing Computation in Text-to-Image Diffusion for Efficient Generation of Image Sets
- **link**: https://arxiv.org/abs/2508.21032
- **venue claim**: ICCV 2025
- **venue evidence** [dblp]: {"venue": "ICCV", "year": "2025", "dblp_title": "Reusing Computation in Text-to-Image Diffusion for Efficient Generation of Image Sets."}
- motivation: Text-to-image diffusion models enable high-quality image generation but are computationally expensive. While prior work optimizes per-inference efficiency, we explore an orthogonal approach: reducing 
- hypothesis: Our approach significantly reduces compute cost while improving image quality.
- method_sketch: We propose a training-free approach that clusters prompts based on semantic similarity and shares computation in early diffusion steps. By leveraging UnClip's text-to-image prior, we enhance diffusion
- dataset: Prompt Template Dataset
- metrics: VQA Score
- baselines: standard approach
- expected_outcome: Our method seamlessly integrates with existing pipelines, scales with prompt sets, and reduces the environmental and financial burden of large-scale text-to-image generation.

### pub-029  ✅
- **paper**: 3D Priors-Guided Diffusion for Blind Face Restoration
- **link**: https://arxiv.org/abs/2409.00991
- **venue claim**: per arXiv comment: This paper was accepted by ACM MM 2024, and the project page is accessible at: https://github.com/83
- **venue evidence** [arxiv:comment]: This paper was accepted by ACM MM 2024, and the project page is accessible at: https://github.com/838143396/3Diffusion
- motivation: Blind face restoration endeavors to restore a clear face image from a degraded counterpart. Recent approaches employing Generative Adversarial Networks (GANs) as priors have demonstrated remarkable su
- hypothesis: A novel diffusion-based framework by embedding the 3D facial priors as structure and identity constraints into a denoising diffusion process will achieve a balance between realism and fidelity.
- method_sketch: The 3D facial image is reconstructed by a 3D Morphable Model (3DMM) using an initial restored face image that has been processed by a pretrained restoration network. A customized multi-level feature e
- dataset: synthetic and real-world datasets
- metrics: PSNR, SSIM, LPIPS, FID-F, FID-G
- baselines: state-of-the-art algorithms
- expected_outcome: Our network performs favorably against state-of-the-art algorithms on synthetic and real-world datasets for blind face restoration.

### pub-030  ✅
- **paper**: Multiclass Graph-Based Large Margin Classifiers: Unified Approach for Support Vectors and Neural Networks
- **link**: https://arxiv.org/abs/2512.13410
- **venue claim**: IEEE Transactions on Neural Networks and Learning Systems (Volume: 36, Issue: 5, May 2025, Pages: 8307 - 8316)
- **venue evidence** [arxiv:journal_ref]: IEEE Transactions on Neural Networks and Learning Systems (Volume: 36, Issue: 5, May 2025, Pages: 8307 - 8316)
- motivation: While large margin classifiers are originally an outcome of an optimization framework, support vectors (SVs) can be obtained from geometric approaches. This article presents advances in the use of Gab
- hypothesis: Activation functions and support edge (SE)-centered neurons affect the classification, proposing smoother functions and structural SV (SSV)-centered neurons to achieve margins with low probabilities a
- method_sketch: We extend the neural network architecture, which can be trained with backpropagation with a softmax function and a cross-entropy loss, or by solving a system of linear equations. A new subgraph-/dista
- dataset: UCI repository
- metrics: Friedman test
- baselines: previous GG-based classifiers, tree-based models
- expected_outcome: Our method was better than previous GG-based classifiers and statistically equivalent to tree-based models.

### pub-032  ✅
- **paper**: Proficient Graph Neural Network Design by Accumulating Knowledge on Large Language Models
- **link**: https://arxiv.org/abs/2408.06717
- **venue claim**: per arXiv comment: Accepted at WSDM 2026. Title changed from "Computation-friendly graph neural network design by accum
- **venue evidence** [arxiv:comment]: Accepted at WSDM 2026. Title changed from "Computation-friendly graph neural network design by accumulating knowledge on large language models" to "Proficient Graph Neural Network 
- motivation: Achieving proficiency in designing data-aware models remains challenging for existing automated approaches, due to their inefficient construction and application of meta-knowledge.
- hypothesis: DesiGNN can deliver top-5.77% initial model proposals for unseen datasets within seconds and achieve consistently superior performance with minimal search cost compared to baselines.
- method_sketch: We propose DesiGNN, a knowledge-centered framework that systematically converts past model design experience into structured, fine-grained knowledge priors well-suited for meta-learning with LLMs. Des
- dataset: NAS-Bench-Graph
- metrics: top 5.77%
- baselines: baselines
- expected_outcome: DesiGNN can deliver top-5.77% initial model proposals for unseen datasets within seconds and achieve consistently superior performance with minimal search cost compared to baselines.

### pub-033  ✅
- **paper**: SHARP: Social Harm Analysis via Risk Profiles for Measuring Inequities in Large Language Models
- **link**: https://arxiv.org/abs/2601.21235
- **venue claim**: Open MIND 2026
- **venue evidence** [openalex]: {"venue": "Open MIND", "year": "2026", "oa_title": "SHARP: Social Harm Analysis via Risk Profiles for Measuring Inequities in Large Language Models"}
- motivation: Large language models (LLMs) are increasingly deployed in high-stakes domains, where rare but severe failures can result in irreversible harm. However, prevailing evaluation benchmarks often reduce co
- hypothesis: SHARP, a framework for multidimensional, distribution-aware evaluation of social harm, can reveal more about the tail exposure and volatility of models with similar average risk.
- method_sketch: This paper introduces Social Harm Analysis via Risk Profiles (SHARP), a framework for multidimensional, distribution-aware evaluation of social harm. SHARP models harm as a multivariate random variabl
- dataset: a fixed corpus of n=901 socially sensitive prompts
- metrics: Conditional Value at Risk (CVaR95)
- baselines: prevailing evaluation benchmarks
- expected_outcome: Models with similar average risk can exhibit more than twofold differences in tail exposure and volatility.

### pub-039  ✅
- **paper**: A Plan Reuse Mechanism for LLM-Driven Agent
- **link**: https://arxiv.org/abs/2512.21309
- **venue claim**: per arXiv comment: This paper is an English version of A Plan Reuse Mechanism for LLM-Driven Agent published in 2024 in
- **venue evidence** [arxiv:comment]: This paper is an English version of A Plan Reuse Mechanism for LLM-Driven Agent published in 2024 in the Journal of Computer Research and Development
- motivation: Integrating large language models (LLMs) into personal assistants enhances their ability to interact with humans, solve complex tasks, and manage IoT devices, but the latency for generating a plan wit
- hypothesis: A plan reuse mechanism for LLM-driven agents can reduce latency by reusing previously generated plans for similar requests.
- method_sketch: We present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse, which leverages the similarities and differences among requests' semantics and uses intent classification to ev
- dataset: real-world dataset
- metrics: effective plan reuse rate, F1 score, accuracy
- baselines: baselines without using the reuse mechanism
- expected_outcome: AgentReuse achieves a high effective plan reuse rate, high F1 score and accuracy in evaluating request similarities, and reduces latency compared to baselines.

### pub-040  ✅
- **paper**: ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Language Models
- **link**: https://arxiv.org/abs/2510.00071
- **venue claim**: per arXiv comment: Accepted by 39th NeurIPS - Foundations of Reasoning in Language Models
- **venue evidence** [arxiv:comment]: Accepted by 39th NeurIPS - Foundations of Reasoning in Language Models
- motivation: Large Reasoning Language Models (LRLMs or LRMs) demonstrate remarkable capabilities in complex reasoning tasks, but suffer from significant computational inefficiencies due to overthinking phenomena.
- hypothesis: Adaptive Reasoning Suppression (ARS), a novel training-free approach that dynamically suppresses redundant reasoning steps while preserving accuracy through adaptive certainty monitoring.
- method_sketch: ARS introduces a multi-checkpoint certainty estimation mechanism with progressive suppression thresholds.
- dataset: mathematical reasoning benchmarks
- metrics: token, latency, energy reduction, accuracy
- baselines: static suppression methods
- expected_outcome: ARS achieves superior efficiency compared to static suppression methods

### pub-041  ✅
- **paper**: IGMiRAG: Intuition-Guided Retrieval-Augmented Generation with Adaptive Mining of In-Depth Memory
- **link**: https://arxiv.org/abs/2602.07525
- **venue claim**: Open MIND 2026
- **venue evidence** [openalex]: {"venue": "Open MIND", "year": "2026", "oa_title": "IGMiRAG: Intuition-Guided Retrieval-Augmented Generation with Adaptive Mining of In-Depth Memory"}
- motivation: Retrieval-augmented generation (RAG) equips large language models (LLMs) with reliable knowledge memory, but recent research integrating graphs and hypergraphs into RAG has misaligned memory organizat
- hypothesis: IGMiRAG, a framework inspired by human intuition-guided reasoning, constructs a hierarchical heterogeneous hypergraph to align multi-granular knowledge, incorporating deductive pathways to simulate re
- method_sketch: IGMiRAG constructs a hierarchical heterogeneous hypergraph to align multi-granular knowledge, incorporating deductive pathways to simulate realistic memory structures. During querying, IGMiRAG distill
- dataset: PopQA
- metrics: EM, F1
- baselines: state-of-the-art
- expected_outcome: IGMiRAG outperforms the state-of-the-art baseline in EM and F1, with token costs adapting to task complexity.

### pub-042  ✅
- **paper**: Large Language Model Evaluation Via Multi AI Agents: Preliminary results
- **link**: https://arxiv.org/abs/2404.01023
- **venue claim**: ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024
- **venue evidence** [arxiv:journal_ref]: ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024
- motivation: As Large Language Models (LLMs) have become integral to both research and daily operations, rigorous evaluation is crucial. This assessment is important not only for individual tasks but also for unde
- hypothesis: A multi-agent AI model can assess and compare the performance of various LLMs.
- method_sketch: We introduce a novel multi-agent AI model that aims to assess and compare the performance of various LLMs. Our model consists of eight distinct AI agents, each responsible for retrieving code based on
- dataset: HumanEval benchmark
- metrics: performance of generated code
- baselines: GPT-3.5, GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, Google Bard, LLAMA, Hugging Face
- expected_outcome: The model will provide insights into the respective capabilities and efficiencies of different LLMs.

### pub-045  ✅
- **paper**: Improving Variable-Length Generation in Diffusion Language Models via Length Regularization
- **link**: https://arxiv.org/abs/2602.07546
- **venue claim**: Open MIND 2026
- **venue evidence** [openalex]: {"venue": "Open MIND", "year": "2026", "oa_title": "Improving Variable-Length Generation in Diffusion Language Models via Length Regularization"}
- motivation: Diffusion Large Language Models (DLLMs) are inherently ill-suited for variable-length generation, as their inference is defined on a fixed-length canvas and implicitly assumes a known target length. W
- hypothesis: This failure arises from an intrinsic length-induced bias in generation confidence estimates, leaving existing DLLMs without a robust way to determine generation length and making variable-length infe
- method_sketch: To address this issue, we propose LR-DLLM, a length-regularized inference framework for DLLMs that treats generation length as an explicit variable and achieves reliable length determination at infere
- dataset: HumanEvalInfilling, McEval
- metrics: Pass@1
- baselines: DreamOn
- expected_outcome: LR-DLLM enables dynamic expansion or contraction of the generation span without modifying the underlying DLLM or its training procedure.

### pub-046  ✅
- **paper**: MirrorDiffusion: Stabilizing Diffusion Process in Zero-shot Image Translation by Prompts Redescription and Beyond
- **link**: https://arxiv.org/abs/2401.03221
- **venue claim**: IEEE Signal Processing Letters 2024
- **venue evidence** [semantic_scholar]: {"venue": "IEEE Signal Processing Letters", "year": "2024", "s2_title": "MirrorDiffusion: Stabilizing Diffusion Process in Zero-Shot Image Translation by Prompts Redescription and 
- motivation: The sampling and inversion processes of Denoising Diffusion Probabilistic Models (DDPM) are stochastic, and thus the inversion process often fail to reconstruct the input content. Specifically, the di
- hypothesis: A prompt redescription mechanism is investigated to align the text prompts with latent code at each time step of the Denoising Diffusion Implicit Models (DDIM) inversion to pursue a structure-preservi
- method_sketch: We propose a prompt redescription strategy to realize a mirror effect between the source and reconstructed image in the diffusion model (MirrorDiffusion). More specifically, MirrorDiffusion is able to
- dataset: LAION-5B dataset
- metrics: CLIP-ACC, Structure Dist, Structure Similarity Index Measure, Learned Perceptual Image Patch Similarity
- baselines: state-of-the-art methods
- expected_outcome: MirrorDiffusion achieves superior performance over the state-of-the-art methods on zero-shot image translation benchmarks by clear margins and practical model stability.

### pub-047  ✅
- **paper**: SSP-IR: Semantic and Structure Priors for Diffusion-based Realistic Image Restoration
- **link**: https://arxiv.org/abs/2407.03635
- **venue claim**: Y. Zhang, H. Zhang, Z. Cheng, R. Xie, L. Song and W. Zhang, "SSP-IR: Semantic and Structure Priors for Diffusion-based R
- **venue evidence** [arxiv:journal_ref]: Y. Zhang, H. Zhang, Z. Cheng, R. Xie, L. Song and W. Zhang, "SSP-IR: Semantic and Structure Priors for Diffusion-based Realistic Image Restoration," in IEEE Transactions on Circuit
- motivation: Realistic image restoration is a crucial task in computer vision, and diffusion-based models for image restoration have garnered significant attention due to their ability to produce realistic results
- hypothesis: A novel image restoration method, SSP-IR, that fully exploits semantic and structure priors from low-quality images to guide the diffusion model in generating semantically faithful and structurally ac
- method_sketch: We integrate the visual comprehension capabilities of Multimodal Large Language Models (explicit) and the visual representations of the original image (implicit) to acquire accurate semantic prior. To
- dataset: synthetic and real-world datasets
- metrics: CLIP similarity, SSIM, CLIPIQA
- baselines: state-of-the-art methods
- expected_outcome: Our method will outperform other state-of-the-art methods overall on both synthetic and real-world datasets.

### pub-048  ✅
- **paper**: Graph Neural Networks Based Analog Circuit Link Prediction
- **link**: https://arxiv.org/abs/2504.10240
- **venue claim**: Pan, G., Zhou, T., Zhao, J., Li, Z., Lin, Y., Ma, B., ... & Wang, S. (2026). Graph Neural Networks Based Analog Circuit
- **venue evidence** [arxiv:journal_ref]: Pan, G., Zhou, T., Zhao, J., Li, Z., Lin, Y., Ma, B., ... & Wang, S. (2026). Graph Neural Networks Based Analog Circuit Link Prediction. Engineering Applications of Artificial Inte
- motivation: Circuit link prediction is crucial in analog circuit design automation, but existing methods face challenges such as insufficient use of topological patterns, data scarcity, and limited adaptability t
- hypothesis: Graph Neural Networks Based Analog Circuit Link Prediction (GNN-ACLP) can tackle these challenges and improve prediction accuracy.
- method_sketch: We introduce the SEAL framework for port-level accuracy, propose Netlist Babel Fish for netlist format conversion, and build a comprehensive dataset SpiceNetlist. Experiments are conducted for intra-d
- dataset: SpiceNetlist, Image2Net, Masala-CHAI
- metrics: accuracy
- baselines: existing methods
- expected_outcome: Improvements in accuracy compared to the baseline and robust feature transfer capabilities in cross-dataset evaluation.

### pub-049  ✅
- **paper**: Distance-Misaligned Training in Graph Transformers and Adaptive Graph-Aware Control
- **link**: https://arxiv.org/abs/2604.22413
- **venue claim**: per arXiv comment: Accepted by Graph Signal Processing Workshop 2026 as an extended abstract
- **venue evidence** [arxiv:comment]: Accepted by Graph Signal Processing Workshop 2026 as an extended abstract
- motivation: Graph Transformers can mix information globally, but this flexibility also creates failure modes: some tasks require long-range communication while others are better served by local interaction.
- hypothesis: The preferred graph-distance bias changes systematically with task locality.
- method_sketch: We study this through a synthetic node-classification benchmark on contextual stochastic block model graphs, where labels are generated by a controllable mixture of local and far-shell signals. We def
- dataset: contextual stochastic block model graphs
- metrics: test accuracy
- baselines: neutral baseline
- expected_outcome: An oracle adaptive controller, given offline access to the task-side distance target, nearly matches the best fixed bias across regimes and strongly improves over a neutral baseline on mixed and local

### pub-050  ✅
- **paper**: Graph neural network for colliding particles with an application to sea ice floe modeling
- **link**: https://arxiv.org/abs/2602.16213
- **venue claim**: The Arabian journal for science and engineering 2026
- **venue evidence** [semantic_scholar]: {"venue": "The Arabian journal for science and engineering", "year": "2026", "s2_title": "Graph neural network for colliding particles with an application to sea ice floe modeling"
- motivation: This paper introduces a novel approach to sea ice modeling using Graph Neural Networks (GNNs), utilizing the natural graph structure of sea ice, where nodes represent individual ice pieces, and edges 
- hypothesis: By utilizing GNNs, the proposed model, termed the Collision-captured Network (CN), integrates data assimilation (DA) techniques to effectively learn and predict sea ice dynamics under various conditio
- method_sketch: The approach was validated using synthetic data, both with and without observed data points, and it was found that the model accelerates the simulation of trajectories without compromising accuracy.
- dataset: synthetic data
- metrics: root mean squared error (RMSE)
- baselines: Traditional numerical methods
- expected_outcome: This advancement offers a more efficient tool for forecasting in marginal ice zones (MIZ) and highlights the potential of combining machine learning with data assimilation for more effective and effic
