# Every date on the timeline, and where it came from

Generated from [`site/data.js`](site/data.js) by [`tools/gen_date_table.py`](tools/gen_date_table.py) — the site and this table cannot drift apart.

**39 mechanisms.** 24 were covered in Session 8; 15 are additions. 12 carry a date caveat, written out in full in the app.

Dates are the **v1 arXiv submission timestamp** (not the announcement date, not the conference publication date, not the version you happen to be reading) or, for things that were never papers, the official release. Where those disagree, the disagreement is recorded rather than resolved silently.

| Date (v1) | Mechanism | Origin | Source used for the date | Caveat |
|---|---|---|---|---|
| `1 Sep 2014` | **Additive** | Bahdanau, Cho, Bengio | [arXiv:1409.0473 v1](https://arxiv.org/abs/1409.0473) | — |
| `17 Aug 2015` | **Dot-product** | Luong, Pham, Manning | [arXiv:1508.04025 v1](https://arxiv.org/abs/1508.04025) | — |
| `8 May 2017` | **Learned APE** | Gehring, Auli, Grangier, Yarats, Dauphin (ConvS2S) | [arXiv:1705.03122 v1](https://arxiv.org/abs/1705.03122) <br> [also evaluated in arXiv:1706.03762 §3.5](https://arxiv.org/abs/1706.03762) | ⚠ see the entry in the app |
| `12 Jun 2017` | **Sinusoidal** | Vaswani et al., §3.5 | [arXiv:1706.03762 v1](https://arxiv.org/abs/1706.03762) | — |
| `12 Jun 2017` | **Vanilla / MHA** | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin | [arXiv:1706.03762 v1](https://arxiv.org/abs/1706.03762) | ⚠ see the entry in the app |
| `6 Mar 2018` | **Relative PE** | Shaw, Uszkoreit, Vaswani | [arXiv:1803.02155 v1](https://arxiv.org/abs/1803.02155) | — |
| `9 Jan 2019` | **Transformer-XL** | Dai, Yang, Yang, Carbonell, Le, Salakhutdinov | [arXiv:1901.02860 v1](https://arxiv.org/abs/1901.02860) | — |
| `23 Apr 2019` | **Sparse Transformer** | Child, Gray, Radford, Sutskever (OpenAI) | [arXiv:1904.10509 v1](https://arxiv.org/abs/1904.10509) | — |
| `6 Nov 2019` | **MQA** | Noam Shazeer | [arXiv:1911.02150 v1](https://arxiv.org/abs/1911.02150) | ⚠ see the entry in the app |
| `25 Dec 2019` | **Top-k** | Zhao, Lin, Zhao, Wang, Sui, Li | [arXiv:1912.11637 v1](https://arxiv.org/abs/1912.11637) | — |
| `13 Jan 2020` | **Reformer** | Kitaev, Kaiser, Levskaya | [arXiv:2001.04451 v1](https://arxiv.org/abs/2001.04451) | — |
| `10 Apr 2020` | **SWA** | Beltagy, Peters, Cohan (Longformer) | [arXiv:2004.05150 v1](https://arxiv.org/abs/2004.05150) <br> [decoder-scale deployment: Mistral 7B, arXiv:2310.06825 (10 Oct 2023)](https://arxiv.org/abs/2310.06825) | ⚠ see the entry in the app |
| `29 Jun 2020` | **Linear attention** | Katharopoulos, Vyas, Pappas, Fleuret | [arXiv:2006.16236 v1](https://arxiv.org/abs/2006.16236) | ⚠ see the entry in the app |
| `28 Jul 2020` | **BigBird** | Zaheer et al. (Google) | [arXiv:2007.14062 v1](https://arxiv.org/abs/2007.14062) | — |
| `30 Sep 2020` | **Performer** | Choromanski et al. (Google) | [arXiv:2009.14794 v1](https://arxiv.org/abs/2009.14794) | — |
| `22 Feb 2021` | **Delta rule** | Schlag, Irie, Schmidhuber | [arXiv:2102.11174 v1](https://arxiv.org/abs/2102.11174) <br> [parallel training algorithm: arXiv:2406.06484 (10 Jun 2024)](https://arxiv.org/abs/2406.06484) | ⚠ see the entry in the app |
| `20 Apr 2021` | **RoPE** | Su, Lu, Pan, Murtadha, Wen, Liu | [arXiv:2104.09864 v1](https://arxiv.org/abs/2104.09864) | — |
| `27 Aug 2021` | **ALiBi** | Press, Smith, Lewis | [arXiv:2108.12409 v1](https://arxiv.org/abs/2108.12409) | — |
| `27 May 2022` | **FlashAttention** | Dao, Fu, Ermon, Rudra, Ré | [arXiv:2205.14135 v1](https://arxiv.org/abs/2205.14135) <br> [FlashAttention-2: arXiv:2307.08691 (17 Jul 2023)](https://arxiv.org/abs/2307.08691) | — |
| `22 May 2023` | **GQA** | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai | [arXiv:2305.13245 v1](https://arxiv.org/abs/2305.13245) | — |
| `27 Jun 2023` | **PI** | Chen, Wong, Chen, Tian (Meta) | [arXiv:2306.15595 v1](https://arxiv.org/abs/2306.15595) | — |
| `late Jun 2023` | **NTK-aware** | bloc97 (Reddit, r/LocalLLaMA) | [r/LocalLLaMA post by bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) <br> [earliest independent timestamp: HF text-generation-inference issue #512, opened 30 Jun 2023](https://github.com/huggingface/text-generation-inference/issues/512) <br> [described and credited in YaRN §3.2, arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | ⚠ see the entry in the app |
| `31 Aug 2023` | **YaRN** | Peng, Quesnelle, Fan, Shippole | [arXiv:2309.00071 v1](https://arxiv.org/abs/2309.00071) | ⚠ see the entry in the app |
| `12 Sep 2023` | **PagedAttention** | Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica | [arXiv:2309.06180 v1](https://arxiv.org/abs/2309.06180) | — |
| `29 Sep 2023` | **Attention sinks** | Xiao, Tian, Chen, Han, Lewis | [arXiv:2309.17453 v1](https://arxiv.org/abs/2309.17453) <br> [related earlier observation: E. Miller, 'Attention Is Off By One', 24 Jul 2023](https://www.evanmiller.org/attention-is-off-by-one.html) <br> [trained per-head sink logit: gpt-oss model card, 5 Aug 2025](https://arxiv.org/abs/2508.10925) | — |
| `21 Feb 2024` | **LongRoPE** | Ding, Zhang, Zhang, Xu, Xia, Zhang, Yang (Microsoft) | [arXiv:2402.13753 v1](https://arxiv.org/abs/2402.13753) | — |
| `10 Apr 2024` | **Infini-attention** | Munkhdalai, Faruqui, Gopal (Google) | [arXiv:2404.07143 v1](https://arxiv.org/abs/2404.07143) | — |
| `7 May 2024` | **MLA** | DeepSeek-AI (DeepSeek-V2) | [arXiv:2405.04434 v1](https://arxiv.org/abs/2405.04434) | — |
| `10 Jun 2024` | **Parallel DeltaNet** | Yang, Wang, Zhang, Kim, Cui, Kim | [arXiv:2406.06484 v1](https://arxiv.org/abs/2406.06484) | — |
| `11 Sep 2024` | **GSA** | Zhang, Yang, Zhu, Qin, Sun, Zhang et al. | [arXiv:2409.07146 v1](https://arxiv.org/abs/2409.07146) | — |
| `7 Oct 2024` | **DIFF Transformer** | Ye, Dong, Xia, Sun, Zhu, Huang, Wei (Microsoft) | [arXiv:2410.05258 v1](https://arxiv.org/abs/2410.05258) | — |
| `9 Dec 2024` | **Gated DeltaNet** | Yang, Kautz, Hatamizadeh (NVIDIA) | [arXiv:2412.06464 v1](https://arxiv.org/abs/2412.06464) | — |
| `16 Feb 2025` | **NSA** | Yuan, Gao, Dai, Luo, Zhao, Zhang, Wu et al. (DeepSeek) | [arXiv:2502.11089 v1](https://arxiv.org/abs/2502.11089) | — |
| `18 Feb 2025` | **MoBA** | Lu, Jiang, Chen, Sun et al. (Moonshot AI) | [arXiv:2502.13189 v1](https://arxiv.org/abs/2502.13189) | ⚠ see the entry in the app |
| `11 Sep 2025` | **Hybrid schedules** | Qwen team (Qwen3-Next); pattern also in Jamba, Samba, MiniMax-01, Nemotron | [vLLM day-0 support post, 11 Sep 2025](https://vllm.ai/blog/2025-09-11-qwen3-next) <br> [hybrid design established in Gated DeltaNet, arXiv:2412.06464](https://arxiv.org/abs/2412.06464) | ⚠ see the entry in the app |
| `29 Sep 2025` | **DSA** | DeepSeek-AI (DeepSeek-V3.2-Exp) | [DeepSeek-V3.2-Exp release, 29 Sep 2025](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) <br> [vLLM day-0 post, 29 Sep 2025](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) <br> [DeepSeek-V3.2 paper, arXiv:2512.02556 (2 Dec 2025)](https://arxiv.org/abs/2512.02556) | — |
| `30 Oct 2025` | **KDA** | Moonshot AI (Kimi Linear) | [arXiv:2510.26692 v1](https://arxiv.org/abs/2510.26692) | — |
| `13 Dec 2025` | **DroPE** | Gelberg, Eguchi, Akiba, Cetin (Sakana AI) | [arXiv:2512.12167 v1](https://arxiv.org/abs/2512.12167) <br> [Sakana AI project page](https://sakana.ai/drope/) <br> [precursor: NoPE, arXiv:2305.19466 (31 May 2023)](https://arxiv.org/abs/2305.19466) | ⚠ see the entry in the app |
| `26 Apr 2026` | **CSA + HCA** | DeepSeek-AI (DeepSeek-V4) | [arXiv:2606.19348 v1, submitted 26 Apr 2026](https://arxiv.org/abs/2606.19348) <br> [DeepSeek-V4 preview release, 24 Apr 2026](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | ⚠ see the entry in the app |
