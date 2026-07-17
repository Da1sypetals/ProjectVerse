# 使用 ffmpeg-the-third 替换音频加载逻辑

## 目标

`sovits-svc-mlx` 使用 `ffmpeg-the-third` 作为唯一输入音频加载实现。FFmpeg 负责容器探测、stream 选择、packet 解码以及 sample format 转换，音频加载模块直接返回包含 MLX Array 的 `Audio`。

本次替换只处理输入音频加载。Input gain、MLX 重采样、切片、特征提取、模型推理、loudness envelope 和 WAV 输出保持现有职责与数值逻辑。

## 对外数据类型

音频加载函数继续返回项目类型 `Audio`：

```rust
pub struct Audio {
    pub samples: mlx_rs::Array,
    pub sample_rate: u32,
}
```

返回值必须满足：

- `samples` dtype 为 `Float32`；
- `samples` shape 为 `[1, sample_count]`；
- `samples` 为 mono；
- `sample_rate` 保留输入音频的原始采样率；
- 音频加载过程不执行响度归一化、gain、clipping 或模型采样率转换；
- 空音频直接报错；
- 仅接受 44.1 kHz 和 48 kHz，其他采样率在加载边界直接报错。

推理代码只依赖 `Audio`，不得依赖 FFmpeg 的 format context、decoder、packet、frame、channel layout 或 AVIO 类型。

## 对外加载接口

加载模块提供两个入口：

```rust
pub fn load_audio(path: impl AsRef<Path>) -> anyhow::Result<Audio>;

pub fn load_audio_bytes(
    bytes: &[u8],
    file_extension: &str,
    mime_type: &str,
) -> anyhow::Result<Audio>;
```

接口名称不得包含 `first_channel`，因为多声道输入需要计算所有声道的平均值。CLI 和 Web server 使用相同的加载 facade。

文件入口直接通过 FFmpeg 打开路径。内存入口不得先写入临时文件，必须使用 seekable custom AVIO 读取上传内容。文件扩展名和 MIME type 只作为 format probing 的提示；实际容器和 codec 由 FFmpeg 探测。

## 模块组织

保留 `audio` 作为推理层可见的 facade，只替换其内部加载实现：

- `src/audio.rs`
  - 定义 `Audio`；
  - 暴露 `load_audio` 和 `load_audio_bytes`；
  - 保留现有 WAV 输出、loudness envelope 和 MLX resampler；
  - 不包含 FFmpeg 解码循环和 AVIO callback。

- `src/audio/decode.rs`
  - 使用 `ffmpeg-the-third` 实现文件与内存输入的统一解码流程；
  - 选择 audio stream；
  - 创建并驱动 decoder；
  - 使用 FFmpeg software resampling 将任意 decoder sample format 转为 packed `Float32`；
  - 将 decoded frame 转换为 MLX Array；
  - 使用 MLX 完成声道平均和 frame 拼接；
  - 返回 `Audio`。

- `src/audio/avio.rs`
  - 持有上传字节、当前位置和长度；
  - 实现 FFmpeg read callback 与 seek callback；
  - 创建和释放 `AVIOContext`；
  - 将 custom AVIO 安装到 `AVFormatContext`；
  - 集中管理该功能需要的 `unsafe`；
  - 保证 callback state、AVIO buffer、`AVIOContext` 和 `AVFormatContext` 的释放顺序正确。

`decode.rs` 和 `avio.rs` 都保持为 `audio` 的私有模块。Web server 不得直接操作 AVIO，推理模块不得直接调用 decoder。

## ffmpeg-the-third 依赖

通过 `cargo add` 添加 `ffmpeg-the-third`。关闭无关的默认 feature，只启用输入加载所需能力：

- `codec`；
- `format`；
- `software-resampling`；
- `build`，由 crate 支持的构建流程编译并静态链接 FFmpeg。

`build` feature 已包含 `static`。不得修改或 vendor `ffmpeg-the-third`、`ffmpeg-sys-the-third` 或 FFmpeg 的源代码。不得使用本地 dependency patch。

最终产物不得要求目标机器预装 FFmpeg 动态库，也不得在运行时调用 `ffmpeg` 命令行程序。

FFmpeg 构建必须包含产品接受格式所需的 demuxer 和 decoder。M4A 所需的 MOV/MP4 demuxer、AAC decoder 和 ALAC decoder 必须包含在静态构建中。支持范围以最终链接进产物的 FFmpeg 能力为准。

输入加载不得依赖额外 decoder 或 fallback 链路。不得保留未使用依赖。

## 文件输入

文件加载流程固定为：

1. 使用 `ffmpeg_the_third::format::input` 打开路径并完成 stream info 探测。
2. 使用 `streams().best(media::Type::Audio)` 选择 FFmpeg 判断的最佳 audio stream。
3. 从该 stream 的 codec parameters 创建 decoder context 并打开 audio decoder。
4. 记录 stream index，只向 decoder 发送该 stream 的 packet。
5. 每次发送 packet 后，持续调用 receive-frame，直到 decoder 暂时没有更多 frame。
6. packet 结束后向 decoder 发送 EOF，并持续接收剩余 frame，直到 decoder 返回 EOF。
7. 将每个 frame 交给统一的 frame conversion 路径。
8. 使用 MLX 拼接所有 frame 的 mono Array。
9. 验证采样率、shape 和非空条件后返回 `Audio`。

不得忽略 decoder flush，因为 AAC 等 codec 可能在内部保留延迟帧。

无法读取 packet、无法解码 packet、frame metadata 变化或 flush 失败时直接返回错误。不得跳过 decode error 后继续生成不完整音频。

## 内存输入与 custom AVIO

`ffmpeg-the-third` 的安全 format input API 接受路径或 URL，没有直接接受 Rust byte slice 的入口。内存上传使用该 crate 公开的 FFmpeg FFI 和 `format::context::Input::wrap` 构造输入 context，不修改依赖源码。

custom AVIO 必须提供：

- read callback：从上传字节当前位置复制 FFmpeg 请求的可用数据，并推进位置；
- seek callback：支持 `SEEK_SET`、`SEEK_CUR`、`SEEK_END` 和 FFmpeg 的 size query；
- 明确的 EOF 返回值；
- checked arithmetic，拒绝负位置、越界位置和整数转换溢出；
- 稳定地址的 callback state，生命周期覆盖完整 demux 和 decode 过程；
- 独立的 FFmpeg AVIO buffer；
- 正确设置 custom-IO flag，避免 FFmpeg 按普通文件方式释放自定义 IO；
- 在 `avformat_open_input` 后调用 `avformat_find_stream_info`；
- 通过 `format::context::Input::wrap` 将成功打开的 raw context 交给 crate 的 RAII context 管理；
- 在所有成功和错误路径上正确释放 AVIO buffer、AVIO context、未被 wrap 的 format context 和 callback state。

扩展名和 MIME type 可用于选择或提示 input format。探测失败时直接返回包含扩展名和 MIME type 的错误，不得回退到旧 decoder。

## Sample format 转换

decoder 可能输出整数、浮点、packed 或 planar frame。加载逻辑使用 `ffmpeg_the_third::software::resampling::Context` 统一转换为：

- sample format：packed `F32`；
- channel layout：与 decoded frame 相同；
- sample rate：与 decoded frame 相同。

这里使用 software resampling context 只转换 sample format 和内存布局，不改变 channel count 或 sample rate。

resampling context 的输入定义必须与实际 frame 的 format、channel layout 和 sample rate 一致。首次 frame 创建 context；后续 frame metadata 必须一致。若输入中途合法改变格式，则显式重建 context，并保证所有输出 frame 的采样率和 channel count 与最终拼接契约一致。无法满足一致性时直接报错。

每次 `run` 后处理输出 frame。decoder flush 完成后继续 flush software resampling context，直到没有剩余输出，防止丢失内部缓冲样本。

## 多声道转 mono

不得选择第一条或任意一条声道。

packed `Float32` frame 的内存顺序为 `[sample0_channel0, sample0_channel1, ..., sample1_channel0, ...]`。对每个 converted frame：

1. 从 packed frame data 创建 dtype 为 `Float32`、shape 为 `[frame_sample_count, channel_count]` 的 MLX Array。
2. 使用 MLX `mean` 沿 channel 维计算算术平均，得到 `[frame_sample_count]`。
3. reshape 为 `[1, frame_sample_count]`。

立体声必须严格得到：

```text
mono[t] = (left[t] + right[t]) / 2
```

任意 `C` 声道输入必须严格得到：

```text
mono[t] = sum(input[t, c], c = 0..C) / C
```

不得使用 FFmpeg 自动 downmix 到 mono，因为 FFmpeg 的 channel mixing matrix 可能根据 channel layout 应用不同权重。不得使用 Rust 循环逐 sample 或逐 channel 计算平均值。

## MLX Array 创建与拼接

从 FFmpeg frame 到推理输入的转换必须满足：

- 使用 MLX API 创建每个 frame 的 Array；
- 使用 MLX reduction 完成声道平均；
- 使用 MLX concatenate 沿 sample 维拼接 frame；
- 不创建保存完整音频 sample 的 Rust `Vec<f32>`；
- 不使用 Rust 循环逐 sample 填充 Tensor；
- 所有 shape 和长度转换使用 checked conversion；
- 返回前执行必要的 MLX evaluation，使 FFmpeg frame 释放后 Array 不再依赖其内存。

允许 Rust 循环驱动 packet、decoder frame 和 resampler frame，因为这些是有状态的 FFmpeg API。该循环不得承担可由 MLX tensor operation 完成的 sample 数值处理。

## 与推理的边界

音频加载完成后返回 `Audio`。推理入口接收 `Audio` 或等价的 `&Array` 加 `sample_rate`，随后保持现有处理顺序：

1. Input gain；
2. 必要时使用现有 MLX `SincResampler` 将 48 kHz 转为模型采样率；
3. 其余现有预处理和推理。

Input gain 是加载完成后的第一项信号处理。FFmpeg sample format 转换和多声道算术平均属于得到规范输入表示所必需的加载步骤。

不得让 FFmpeg 将 48 kHz 转为 44.1 kHz。该数值操作继续由现有 MLX resampler 实现。

## 错误处理

所有错误就地返回并带有输入上下文。至少覆盖：

- 文件无法打开；
- 内存输入为空；
- format probing 失败；
- 不存在 audio stream；
- decoder 无法创建或打开；
- packet 读取失败；
- packet 发送失败；
- frame 接收失败；
- decoder flush 失败；
- sample format 转换失败；
- software resampling flush 失败；
- channel count 为零；
- sample rate 不为 44.1 kHz 或 48 kHz；
- frame metadata 不一致；
- 音频没有 decoded sample；
- sample count 超过 MLX shape 支持范围；
- MLX Array 创建、mean、reshape、concatenate 或 evaluation 失败。

不得捕获错误后改用外部 `ffmpeg` 命令或其他 decoder。不得返回部分解码结果。

## 验证要求

实现后必须使用真实音频和真实静态 FFmpeg 构建验证，不得使用 mock。

至少验证：

- mono WAV 44.1 kHz；
- stereo WAV 48 kHz，左右声道使用可独立确认的不同信号；
- AAC in M4A 44.1 kHz；
- AAC in M4A 48 kHz；
- ALAC in M4A；
- MP3；
- FLAC；
- Ogg Vorbis；
- 文件路径输入；
- Web 上传的内存输入；
- 需要 seek 才能完成探测的容器；
- decoder flush 后才输出末尾 sample 的 codec；
- 空文件、损坏容器、无 audio stream、unsupported codec 和不支持采样率的失败路径。

对同一输入，文件路径入口与内存入口必须产生相同的 sample rate、shape 和 MLX sample values。立体声测试必须直接验证每个输出 sample 等于对应左右声道的算术平均，容差只覆盖 `Float32` 运算误差。

完成实现后运行：

- `cargo fmt --check`；
- `cargo check --all-targets`；
- `cargo test --all-targets`；
- release 模式的 CLI 与 Web inference smoke test；
- 检查最终可执行文件的动态库依赖，确认不存在 FFmpeg 动态库依赖。
