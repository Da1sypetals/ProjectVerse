# so-vits-svc Web 推理页面功能规范

## 1. 文档用途

本文档定义 `examples/web.rs` 所提供 HTML 页面的功能要求。目标读者是负责实现、修改或验证该页面及其服务端协议的 LLM Agent。

本文档不定义任何布局、颜色、尺寸、字体、间距、动画或其他视觉要求。

## 2. 功能范围

页面 MUST 完成以下任务：

1. 接收一个本地人声音频文件。
2. 收集 so-vits-svc 推理参数。
3. 允许用户选择 GAN 后是否运行 Refiner，以及使用哪一种 Refiner。
4. 向本地 Web 服务提交推理请求。
5. 接收 WAV 推理结果。
6. 提供结果播放和下载能力。
7. 报告输入错误、参数错误、解码错误、推理错误和服务状态错误。

页面 MUST NOT：

- 要求用户指定输出路径或输出文件名。
- 将音频上传到本地服务以外的第三方服务。
- 在浏览器持久化上传音频或推理结果。
- 依赖 checkpoint manifest。
- 在页面中切换或修改 checkpoint 路径。
- 同时提交多个输入文件。

## 3. 服务端启动依赖

Web 服务启动时 MUST 独立接收以下 checkpoint：

- GAN checkpoint。
- Shallow Diffusion checkpoint。
- Flow Matching checkpoint。
- ContentVec checkpoint。
- FCPE checkpoint。
- NSF-HiFiGAN checkpoint。

服务启动时 MUST 完成全部模型加载。任一必要 checkpoint 加载失败时，服务 MUST fast-fail。

默认监听地址为 `127.0.0.1:3000`。

默认最大 multipart request 大小为 256 MiB。该限制 MUST 可通过服务端 CLI 参数修改。

## 4. HTTP 路由

服务 MUST 提供以下路由：

- `GET /`
  - 返回推理页面。
- `GET /health`
  - 服务正常时返回 HTTP 204。
- `POST /api/infer`
  - 接收 `multipart/form-data` 推理请求。
  - 成功时返回 WAV 数据。

## 5. 音频输入

### 5.1 文件选择

页面 MUST 支持：

- 通过文件选择操作输入文件。
- 通过 drag and drop 输入文件。

选择新文件后 MUST 替换之前选择的文件。

未选择文件时提交推理，页面 MUST：

- 阻止请求。
- 报告缺少输入音频。
- 将交互焦点返回音频输入操作。

### 5.2 支持格式

页面 MUST 将文件原始 bytes、文件扩展名和 MIME type 发送给服务端。

服务端 MUST 使用静态链接的 FFmpeg 解码输入，不得根据页面维护的固定格式白名单限制该 FFmpeg 构建支持的格式。

服务端 MUST：

- 解码为 mono。
- 对多声道输入的所有声道计算逐 sample 算术平均，不得选择单个声道或使用加权 downmix。
- 接受 44.1 kHz 或 48 kHz 输入。
- 将 48 kHz 输入重采样为模型使用的 44.1 kHz。
- 拒绝其他 sample rate。
- 拒绝空文件。
- 拒绝超过配置上传大小的文件。

### 5.3 输入数据生命周期

上传文件和生成结果 MUST 只保存在内存中。

Web 服务 MUST NOT 为单次请求创建临时音频文件。

## 6. 公共推理参数

以下参数对所有 Refiner 模式有效：

### 6.1 `input_gain`

- 类型：整数。
- 单位：dB。
- 默认值：`-2`。
- 有效范围：`-12` 到 `12`，包含边界。
- MUST 在音频解码完成后立即应用。
- MUST 早于重采样、silence slicing、F0 提取、内容特征提取和其他音频处理。
- 线性增益 MUST 按 `10^(input_gain / 20)` 计算。
- MUST NOT 对增益后的音频执行额外 clipping 或 normalization。

### 6.2 `pitch_shift`

- 类型：浮点数。
- 单位：semitone。
- 默认值：`12.0`。
- 页面接受范围：`-48` 到 `48`。
- 页面 MUST 提供将值直接设为 `-12` 和 `+12` 的快捷操作。
- 快捷操作 MUST 设置绝对值，不得基于当前值累加。

### 6.3 `noise_scale`

- 类型：浮点数。
- 默认值：`0.4`。
- MUST 大于等于 `0`。
- 作为 GAN latent noise scale。

### 6.4 `predict_f0`

- 类型：布尔值。
- 默认值：`false`。
- 启用时使用 GAN automatic F0 decoder。

### 6.5 `loudness_envelope_adjustment`

- 类型：浮点数。
- 页面默认值：`1.0`。
- 有效范围：`0.0` 到 `1.0`。
- 控制输入与输出 loudness envelope 的融合强度。

## 7. Refiner 选择

`refiner` MUST 是以下值之一：

- `none`
- `shallow_diffusion`
- `flow_matching`

默认值 MUST 为 `none`。

任意时刻 MUST 只选择一种 Refiner。

页面 MUST 根据当前选择决定哪些参数参与验证和请求：

- `none`
  - 不提交 Shallow Diffusion 专属参数。
  - 不提交 Flow Matching 专属参数。
- `shallow_diffusion`
  - 提交 Shallow Diffusion 专属参数。
  - 不提交 Flow Matching 专属参数。
- `flow_matching`
  - 提交 Flow Matching 专属参数。
  - 不提交 Shallow Diffusion 专属参数。

未启用的 Refiner 参数 MUST 不参与客户端表单校验。

## 8. Shallow Diffusion 参数

以下参数仅在 `refiner=shallow_diffusion` 时有效。

### 8.1 `diffusion_steps`

- 类型：整数。
- 默认值：`100`。
- 最小值：`2`。
- 最大值不得超过 checkpoint diffusion schedule。

### 8.2 `diffusion_speedup`

- 类型：整数。
- 默认值：`10`。
- 最小值：`1`。
- MUST 满足 `diffusion_steps / diffusion_speedup >= 2`。
- 当 `diffusion_steps` 改变时，页面 MUST 将允许的最大值更新为 `floor(diffusion_steps / 2)`。
- 当前值超过新的最大值时，页面 MUST 将其调整到新的最大值。

### 8.3 `second_encoding`

- 类型：布尔值。
- 默认值：`false`。
- 启用时，在 Shallow Diffusion 前重新编码 GAN waveform。
- 该参数 MUST NOT 影响 Flow Matching。

## 9. Flow Matching 参数

以下参数仅在 `refiner=flow_matching` 时有效。

### 9.1 `flow_matching_steps`

- 类型：整数。
- 默认值：`50`。
- 最小值：`1`。
- 表示 endpoint ODE Euler integration step count。

## 10. Silence Slicing

### 10.1 `slicing`

- 类型：布尔值。
- 默认值：`true`。

当 `slicing=false` 时：

- 页面 MUST 不提交 slicing 专属参数。
- slicing 专属参数 MUST 不参与客户端表单校验。
- 服务端 MUST 调用单段推理。

当 `slicing=true` 时：

- 页面 MUST 提交全部 slicing 参数。
- 服务端 MUST 调用 sliced inference。

### 10.2 `threshold_db`

- 类型：浮点数。
- 默认值：`-40.0`。
- 表示 silence threshold。

### 10.3 `padding_seconds`

- 类型：浮点数。
- 默认值：`0.5`。
- MUST 大于等于 `0`。
- 表示非静音切片两侧的上下文 padding。

### 10.4 `clip_seconds`

- 类型：浮点数。
- 默认值：`0.0`。
- MUST 大于等于 `0`。
- `0` 表示不进行强制定长切分。

### 10.5 `crossfade_seconds`

- 类型：浮点数。
- 默认值：`0.0`。
- MUST 大于等于 `0`。
- 表示相邻强制切片的 overlap duration。
- 当 `clip_seconds > 0` 时，MUST 满足 `crossfade_seconds <= clip_seconds`。

### 10.6 `crossfade_ratio`

- 类型：浮点数。
- 默认值：`0.75`。
- 有效范围：`0.0` 到 `1.0`。
- 表示 overlap 中参与 linear crossfade 的比例。

## 11. 请求协议

页面 MUST 使用 `POST /api/infer` 和 `multipart/form-data`。

### 11.1 必须字段

- `audio`
- `input_gain`
- `pitch_shift`
- `noise_scale`
- `loudness_envelope_adjustment`
- `predict_f0`
- `refiner`
- `slicing`

### 11.2 条件字段

当 `refiner=shallow_diffusion`：

- `diffusion_steps`
- `diffusion_speedup`
- `second_encoding`

当 `refiner=flow_matching`：

- `flow_matching_steps`

当 `slicing=true`：

- `threshold_db`
- `padding_seconds`
- `clip_seconds`
- `crossfade_seconds`
- `crossfade_ratio`

### 11.3 服务端解析

服务端 MUST：

- 从默认参数对象开始解析请求。
- 使用请求中存在的字段覆盖默认值。
- 拒绝无法转换为目标类型的字段。
- 拒绝未知 multipart 字段。
- 拒绝缺少字段名的 multipart part。
- 拒绝非法 `refiner` 值。

服务端 MAY 为兼容旧客户端继续接受布尔字段 `shallow_diffusion`：

- `true` 映射为 `refiner=shallow_diffusion`。
- `false` 映射为 `refiner=none`。

## 12. 推理执行

模型推理 MUST 在独立 engine thread 中执行。

Web runtime MUST 通过有界 channel 将请求发送给 engine thread。推理请求 MUST 串行访问同一个模型实例，避免并发修改模型状态。

推理顺序 MUST 为：

1. FFmpeg 音频解码，并使用 MLX 对所有声道计算逐 sample 算术平均。
2. 应用 Input Gain。
3. 可选 silence slicing；启用时，silence detection MUST 使用增益后的音频。
4. 必要的输入重采样。
5. 公共预处理。
6. GAN inference。
7. 根据 `refiner` 执行零个或一个 Refiner。
8. 必要时执行 NSF-HiFiGAN vocoder。
9. 可选 loudness envelope adjustment。
10. 可选 sliced inference 合并。
11. 编码为 WAV response。

## 13. 提交状态

页面 MUST 防止同一表单被重复提交。

请求进行期间 MUST：

- 禁止再次提交。
- 报告推理正在进行。
- 保留提交动作的原始语义。

快速完成的请求 SHOULD 保持至少 400 ms 的可感知 processing 状态，避免状态瞬间闪烁。

请求结束后 MUST 恢复提交能力。

## 14. 成功响应

成功响应 MUST：

- 使用 HTTP 成功状态。
- `Content-Type` 为 `audio/wav`。
- body 为 mono、44.1 kHz、Float32 WAV。
- 包含 `Content-Disposition`。
- 包含 `x-output-samples`。
- 包含 `x-inference-ms`。

### 14.1 输出文件名

输出文件名 MUST 从输入文件名派生，用户不得手动指定。

规则：

1. 移除输入文件的目录部分。
2. 移除最后一个扩展名。
3. 保留其余文件名，包括中间的 `.`。
4. 添加后缀 `-converted.wav`。

示例：

- `voice.demo.mp3` 生成 `voice.demo-converted.wav`。
- `声音.flac` 生成 `声音-converted.wav`。
- `audio` 生成 `audio-converted.wav`。

`Content-Disposition` MUST：

- 提供只包含安全 ASCII 字符的 fallback filename。
- 使用 `filename*=UTF-8''...` 保留原始 Unicode 文件名。

### 14.2 结果功能

页面 MUST：

- 使用 response WAV 创建本地 object URL。
- 支持播放结果。
- 支持下载结果。
- 将下载文件名设置为派生输出文件名。
- 报告输出 sample count 对应的时长。
- 报告服务端推理耗时。
- 报告 response 数据大小。
- 新结果产生时释放之前的 object URL。

## 15. 错误处理

客户端 MUST 处理：

- 未选择文件。
- 浏览器表单约束失败。
- 网络请求失败。
- 服务端非成功响应。
- 无法读取错误 response。

服务端 MUST 区分并返回：

- HTTP 400：multipart、字段、文件名、字段类型或缺少输入错误。
- HTTP 413：请求或上传文件超过大小限制。
- HTTP 422：音频解码、sample rate、推理或输出编码错误。
- HTTP 503：engine thread 或 channel 不可用。

客户端收到服务端错误时 MUST：

- 读取并报告服务端文本错误。
- 提供重新检查音频格式和参数的操作指引。
- 将焦点移动到错误信息或对应的输入操作。
- 保留已选择文件和参数，允许修正后重新提交。

## 16. 可访问性行为

页面功能 MUST 可通过键盘操作。

页面 MUST：

- 为所有输入提供可访问名称。
- 允许通过键盘触发文件选择、Refiner 选择、开关、快捷 pitch shift、提交、播放和下载。
- 使用 live region 报告文件选择、推理进度和错误。
- 提交失败时管理焦点。
- 推理成功时将焦点移动到结果区域。
- 不阻止输入框 paste。
- 不禁用浏览器 zoom。
- 遵守 `prefers-reduced-motion`。

## 17. 状态初始化

页面加载后 MUST 初始化：

- slicing 参数是否参与交互。
- 当前 Refiner 参数是否参与交互。
- `diffusion_speedup` 的最大值。
- pitch shift 快捷值状态。

初始化 MUST 与默认请求参数一致。

## 18. 验收条件

实现或修改页面后 MUST 验证：

1. 未选择音频时不会发送请求。
2. 文件选择和 drag and drop 均可替换当前输入。
3. `pitch_shift` 快捷操作分别设置为 `-12` 和 `+12`。
4. `refiner=none` 请求不包含 Refiner 专属参数。
5. Shallow Diffusion 请求只包含其专属参数。
6. Flow Matching 请求只包含其专属参数。
7. 关闭 slicing 后请求不包含 slicing 专属参数。
8. 三种 Refiner 均能完成真实推理。
9. 44.1 kHz 和 48 kHz FFmpeg 输入均能推理。
10. 非 44.1 kHz/48 kHz 输入被拒绝。
11. 输出是 mono、44.1 kHz、Float32 WAV。
12. 输出文件名由输入文件名派生。
13. Unicode 输入文件名能够通过 `Content-Disposition` 正确下载。
14. 服务端错误会被客户端完整报告。
15. 重复推理会释放旧 object URL。
16. `GET /health` 返回 HTTP 204。
17. Web example 的 Rust tests、release build 和真实 multipart 请求全部通过。
