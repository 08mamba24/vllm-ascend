# 310P MLA 调试信息采集

当 310P 上出现回答重复、`<think>` 模板残留、或无法按 EOS 停止时，可先采集最小定位信息。

## 使用方法

```bash
tools/collect_310p_mla_debug_info.sh --out /tmp/mla_debug.txt
```

如果需要同时检查 tokenizer 的 EOS / special tokens：

```bash
tools/collect_310p_mla_debug_info.sh \
  --model <model_path_or_hf_id> \
  --prompt '你是谁？' \
  --out /tmp/mla_debug.txt
```

如果你有复现脚本（例如 `run_vllm.py`），也建议带上 `--script`，脚本会自动提取
`apply_chat_template` 与 `SamplingParams` 相关关键行：

```bash
tools/collect_310p_mla_debug_info.sh \
  --script run_vllm.py \
  --repo /path/to/vllm-ascend \
  --base-ref origin/main \
  --out /tmp/mla_debug.txt
```

如果采集日志里的 `== Git ==` 为空，可显式指定 `--repo`，尤其是在容器里 `cwd` 不在代码仓根目录时。

## 输出内容

- Git 分支与 commit
- 与 `base-ref`（默认 `origin/main`）相比的新增提交和变更文件清单（用于评估“是否可合入 main”）
- Python / torch / torch_npu / vllm / vllm_ascend 版本
- `VLLM_*`、`ASCEND_*`、`HCCL_*`、`PYTORCH_NPU_*` 等环境变量
- 310P MLA 关键开关值（例如 `VLLM_ASCEND_310P_MLA_KV_CACHE` / `..._FORMAT`）
- 快速健全性检查告警（KV cache 双变量冲突、Ascend 可见卡不一致、`torch +cpu` 提示）
- 建议补充的启动参数与采样参数
- （可选）tokenizer 的 `eos_token_id`、special tokens、prompt 尾部 token ids
- （可选）从复现脚本中提取的模板/采样参数线索

建议将输出文件与最小复现脚本一起提供。

## 改动必要性与后续完善建议

- 必要性：你当前问题同时涉及环境、模板、后端实现和版本漂移，这个采集脚本把这些关键点收拢为一次采集，能显著降低定位回合数。
- 对 main 合入的价值：新增 `--base-ref` 后可以直接看到当前分支相对主线的提交与文件差异，便于快速判断“哪些是调试工具改动，哪些是模型逻辑改动”。
- 建议继续完善：
  1. 增加 `--json` 输出，方便自动化归档与对比。
  2. 增加 `--tokenizer-from-script`，直接运行复现脚本里的模板逻辑而不是手传 prompt。
  3. 增加“必填字段检查”，当缺少 `stop/ignore_eos/first generated token ids` 时给显式提醒。
