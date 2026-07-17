use std::fmt::Display;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::header::{CONTENT_DISPOSITION, CONTENT_TYPE};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use clap::Parser;
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use sovits_svc_mlx::audio::{load_audio_bytes_first_channel, wav_float_bytes};
use sovits_svc_mlx::inference::{InferenceOptions, Refiner, SliceInferenceOptions, SovitsSvc};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

const OUTPUT_SAMPLE_RATE: u32 = 44_100;

#[derive(Debug, Parser)]
#[command(version, about = "Serve the local so-vits-svc MLX inference interface")]
struct Arguments {
    /// Converted GAN checkpoint.
    #[arg(long, default_value = "../ckpt/mlx/gan/opencpop/model.safetensors")]
    gan_checkpoint: PathBuf,

    /// Converted shallow-diffusion checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/refine/shallow_diffusion/guan/model.safetensors"
    )]
    shallow_diffusion_checkpoint: PathBuf,

    /// Converted Flow Matching checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/refine/flow_matching/opencpop/model.safetensors"
    )]
    flow_matching_checkpoint: PathBuf,

    /// Converted ContentVec checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/encoder/contentvec/model.safetensors"
    )]
    contentvec_checkpoint: PathBuf,

    /// FCPE MLX safetensors checkpoint.
    #[arg(long, default_value = "../ckpt/mlx/pitch/fcpe/model.safetensors")]
    fcpe_checkpoint: PathBuf,

    /// Converted NSF-HiFiGAN checkpoint.
    #[arg(
        long,
        default_value = "../ckpt/mlx/vocoder/nsf_hifigan/model.safetensors"
    )]
    vocoder_checkpoint: PathBuf,

    /// Address used by the local web server.
    #[arg(long, default_value = "127.0.0.1:3000")]
    bind: SocketAddr,

    /// Maximum multipart request size in MiB.
    #[arg(long, default_value_t = 256)]
    max_upload_mib: usize,
}

#[derive(Clone)]
struct AppState {
    engine: mpsc::Sender<EngineRequest>,
    max_upload_bytes: usize,
}

struct EngineRequest {
    input: EncodedAudio,
    inference_options: InferenceOptions,
    slice_options: Option<SliceInferenceOptions>,
    reply: oneshot::Sender<std::result::Result<EngineReply, String>>,
}

struct EncodedAudio {
    bytes: Vec<u8>,
    file_extension: String,
    mime_type: String,
}

struct EngineReply {
    wav: Vec<u8>,
    sample_count: i32,
    elapsed_milliseconds: u128,
}

struct WebParameters {
    inference: InferenceOptions,
    slicing: bool,
    slice: SliceInferenceOptions,
}

impl Default for WebParameters {
    fn default() -> Self {
        let mut inference = InferenceOptions::default();
        inference.loudness_envelope_adjustment = 0.0;
        Self {
            inference,
            slicing: true,
            slice: SliceInferenceOptions::default(),
        }
    }
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            [(CONTENT_TYPE, "text/plain; charset=utf-8")],
            self.message,
        )
            .into_response()
    }
}

fn parse_field<T>(name: &str, value: &str) -> std::result::Result<T, ApiError>
where
    T: FromStr,
    T::Err: Display,
{
    value
        .parse::<T>()
        .map_err(|error| ApiError::new(StatusCode::BAD_REQUEST, format!("Invalid {name}: {error}")))
}

fn parse_input_gain(value: &str) -> std::result::Result<i32, ApiError> {
    let input_gain = parse_field("input_gain", value)?;
    if !(-12..=12).contains(&input_gain) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "input_gain must be an integer between -12 and 12 dB",
        ));
    }
    Ok(input_gain)
}

fn output_name(input_name: &str) -> String {
    let basename = input_name.rsplit(['/', '\\']).next().unwrap_or(input_name);
    let stem = Path::new(basename)
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("audio");
    format!("{stem}-converted.wav")
}

fn content_disposition(output_name: &str) -> HeaderValue {
    let mut fallback_stem = Path::new(output_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("audio-converted")
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        .collect::<String>();
    if fallback_stem.is_empty() {
        fallback_stem = "audio-converted".to_owned();
    }
    let fallback = format!("{fallback_stem}.wav");
    let encoded = utf8_percent_encode(output_name, NON_ALPHANUMERIC);
    HeaderValue::from_str(&format!(
        "attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"
    ))
    .expect("derived output name must produce a valid header")
}

fn process_request(model: &mut SovitsSvc, request: EngineRequest) {
    let started = Instant::now();
    let result = (|| -> Result<EngineReply> {
        let input = load_audio_bytes_first_channel(
            &request.input.bytes,
            &request.input.file_extension,
            &request.input.mime_type,
        )?;
        let output = if let Some(slice_options) = request.slice_options {
            model.infer_sliced(
                &input.samples,
                input.sample_rate as i32,
                &request.inference_options,
                &slice_options,
            )?
        } else {
            model
                .infer(
                    &input.samples,
                    input.sample_rate as i32,
                    &request.inference_options,
                )?
                .audio
        };
        let sample_count = output.shape()[1];
        let wav = wav_float_bytes(&output, OUTPUT_SAMPLE_RATE)?;
        Ok(EngineReply {
            wav,
            sample_count,
            elapsed_milliseconds: started.elapsed().as_millis(),
        })
    })()
    .map_err(|error| format!("{error:#}"));
    let _ = request.reply.send(result);
}

fn spawn_engine(
    gan_checkpoint: PathBuf,
    shallow_diffusion_checkpoint: PathBuf,
    flow_matching_checkpoint: PathBuf,
    contentvec_checkpoint: PathBuf,
    fcpe_checkpoint: PathBuf,
    vocoder_checkpoint: PathBuf,
) -> Result<(mpsc::Sender<EngineRequest>, thread::JoinHandle<()>)> {
    let (sender, mut receiver) = mpsc::channel::<EngineRequest>(1);
    let (ready_sender, ready_receiver) =
        std_mpsc::sync_channel::<std::result::Result<(), String>>(1);
    let handle = thread::Builder::new()
        .name("sovits-mlx-inference".to_owned())
        .spawn(move || {
            let model = SovitsSvc::load(
                gan_checkpoint,
                shallow_diffusion_checkpoint,
                flow_matching_checkpoint,
                contentvec_checkpoint,
                fcpe_checkpoint,
                vocoder_checkpoint,
            );
            let mut model = match model {
                Ok(model) => model,
                Err(error) => {
                    let _ = ready_sender.send(Err(format!("{error:#}")));
                    return;
                }
            };
            if ready_sender.send(Ok(())).is_err() {
                return;
            }
            while let Some(request) = receiver.blocking_recv() {
                process_request(&mut model, request);
            }
        })
        .context("failed to start inference thread")?;
    ready_receiver
        .recv()
        .context("inference thread exited during startup")?
        .map_err(anyhow::Error::msg)?;
    Ok((sender, handle))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn infer(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> std::result::Result<Response, ApiError> {
    let mut input = None;
    let mut parameters = WebParameters::default();
    while let Some(field) = multipart.next_field().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("Invalid multipart request: {error}"),
        )
    })? {
        let name = field
            .name()
            .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "Multipart field has no name"))?
            .to_owned();
        if name == "audio" {
            let file_name = field
                .file_name()
                .ok_or_else(|| {
                    ApiError::new(StatusCode::BAD_REQUEST, "Uploaded audio has no filename")
                })?
                .to_owned();
            let mime_type = field.content_type().unwrap_or("").to_owned();
            let file_extension = Path::new(&file_name)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_owned();
            let bytes = field.bytes().await.map_err(|error| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("Failed to read uploaded audio: {error}"),
                )
            })?;
            if bytes.is_empty() {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "The uploaded audio is empty",
                ));
            }
            if bytes.len() > state.max_upload_bytes {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "The uploaded audio exceeds the configured request limit",
                ));
            }
            input = Some((
                EncodedAudio {
                    bytes: bytes.to_vec(),
                    file_extension,
                    mime_type,
                },
                output_name(&file_name),
            ));
            continue;
        }
        let value = field.text().await.map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("Failed to read {name}: {error}"),
            )
        })?;
        match name.as_str() {
            "input_gain" => {
                parameters.inference.input_gain_db = parse_input_gain(&value)?;
            }
            "pitch_shift" => {
                parameters.inference.pitch_shift = parse_field(&name, &value)?;
            }
            "noise_scale" => {
                parameters.inference.noise_scale = parse_field(&name, &value)?;
            }
            "predict_f0" => {
                parameters.inference.predict_f0 = parse_field(&name, &value)?;
            }
            "refiner" => {
                parameters.inference.refiner = match value.as_str() {
                    "none" => Refiner::None,
                    "shallow_diffusion" => Refiner::ShallowDiffusion,
                    "flow_matching" => Refiner::FlowMatching,
                    _ => {
                        return Err(ApiError::new(
                            StatusCode::BAD_REQUEST,
                            format!("Invalid refiner: {value}"),
                        ));
                    }
                };
            }
            "shallow_diffusion" => {
                parameters.inference.refiner = if parse_field(&name, &value)? {
                    Refiner::ShallowDiffusion
                } else {
                    Refiner::None
                };
            }
            "diffusion_steps" => {
                parameters.inference.diffusion_steps = parse_field(&name, &value)?;
            }
            "diffusion_speedup" => {
                parameters.inference.diffusion_speedup = parse_field(&name, &value)?;
            }
            "flow_matching_steps" => {
                parameters.inference.flow_matching_steps = parse_field(&name, &value)?;
            }
            "loudness_envelope_adjustment" => {
                parameters.inference.loudness_envelope_adjustment = parse_field(&name, &value)?;
            }
            "second_encoding" => {
                parameters.inference.second_encoding = parse_field(&name, &value)?;
            }
            "slicing" => {
                parameters.slicing = parse_field(&name, &value)?;
            }
            "threshold_db" => {
                parameters.slice.threshold_db = parse_field(&name, &value)?;
            }
            "padding_seconds" => {
                parameters.slice.padding_seconds = parse_field(&name, &value)?;
            }
            "clip_seconds" => {
                parameters.slice.clip_seconds = parse_field(&name, &value)?;
            }
            "crossfade_seconds" => {
                parameters.slice.crossfade_seconds = parse_field(&name, &value)?;
            }
            "crossfade_ratio" => {
                parameters.slice.crossfade_ratio = parse_field(&name, &value)?;
            }
            _ => {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("Unknown multipart field: {name}"),
                ));
            }
        }
    }

    let (input, output_name) =
        input.ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "Missing input audio"))?;
    let (reply_sender, reply_receiver) = oneshot::channel();
    state
        .engine
        .send(EngineRequest {
            input,
            inference_options: parameters.inference,
            slice_options: parameters.slicing.then_some(parameters.slice),
            reply: reply_sender,
        })
        .await
        .map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "The inference engine is unavailable",
            )
        })?;
    let reply = reply_receiver
        .await
        .map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "The inference engine stopped before completing the request",
            )
        })?
        .map_err(|message| ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, message))?;

    let mut response = reply.wav.into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response
        .headers_mut()
        .insert(CONTENT_DISPOSITION, content_disposition(&output_name));
    response.headers_mut().insert(
        HeaderName::from_static("x-output-samples"),
        HeaderValue::from_str(&reply.sample_count.to_string())
            .expect("sample count must be a valid header"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-inference-ms"),
        HeaderValue::from_str(&reply.elapsed_milliseconds.to_string())
            .expect("elapsed time must be a valid header"),
    );
    Ok(response)
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
}

async fn run_server(
    bind: SocketAddr,
    engine: mpsc::Sender<EngineRequest>,
    max_upload_bytes: usize,
) -> Result<()> {
    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/api/infer", post(infer))
        .with_state(AppState {
            engine,
            max_upload_bytes,
        })
        .layer(DefaultBodyLimit::max(max_upload_bytes));
    let listener = TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind web server to {bind}"))?;
    println!("web interface: http://{bind}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("web server failed")
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let max_upload_bytes = arguments
        .max_upload_mib
        .checked_mul(1024 * 1024)
        .context("maximum upload size is too large")?;
    let (engine, engine_thread) = spawn_engine(
        arguments.gan_checkpoint,
        arguments.shallow_diffusion_checkpoint,
        arguments.flow_matching_checkpoint,
        arguments.contentvec_checkpoint,
        arguments.fcpe_checkpoint,
        arguments.vocoder_checkpoint,
    )?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create web runtime")?;
    let server_result = runtime.block_on(run_server(arguments.bind, engine, max_upload_bytes));
    drop(runtime);
    engine_thread
        .join()
        .map_err(|_| anyhow::anyhow!("inference thread panicked"))?;
    server_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn web_defaults_match_reference_interfaces() {
        let parameters = WebParameters::default();
        assert_eq!(parameters.inference.input_gain_db, 0);
        assert_eq!(parameters.inference.pitch_shift, 0.0);
        assert_eq!(parameters.inference.noise_scale, 0.4);
        assert!(!parameters.inference.predict_f0);
        assert_eq!(parameters.inference.refiner, Refiner::None);
        assert_eq!(parameters.inference.diffusion_steps, 100);
        assert_eq!(parameters.inference.diffusion_speedup, 10);
        assert_eq!(parameters.inference.flow_matching_steps, 50);
        assert_eq!(parameters.inference.loudness_envelope_adjustment, 0.0);
        assert!(!parameters.inference.second_encoding);
        assert!(parameters.slicing);
        assert_eq!(parameters.slice.threshold_db, -40.0);
        assert_eq!(parameters.slice.padding_seconds, 0.5);
        assert_eq!(parameters.slice.clip_seconds, 0.0);
        assert_eq!(parameters.slice.crossfade_seconds, 0.0);
        assert_eq!(parameters.slice.crossfade_ratio, 0.75);
    }

    #[test]
    fn output_name_is_derived_from_input_name() {
        assert_eq!(output_name("voice.demo.mp3"), "voice.demo-converted.wav");
        assert_eq!(output_name("声音.flac"), "声音-converted.wav");
        assert_eq!(output_name("audio"), "audio-converted.wav");
    }

    #[test]
    fn input_gain_accepts_only_integer_values_in_range() {
        assert_eq!(parse_input_gain("-12").unwrap(), -12);
        assert_eq!(parse_input_gain("0").unwrap(), 0);
        assert_eq!(parse_input_gain("12").unwrap(), 12);
        assert!(parse_input_gain("-13").is_err());
        assert!(parse_input_gain("13").is_err());
        assert!(parse_input_gain("1.5").is_err());
    }

    #[test]
    fn checkpoint_defaults_use_role_directories() {
        let arguments = Arguments::try_parse_from(["web"]).unwrap();
        assert_eq!(
            arguments.gan_checkpoint,
            PathBuf::from("../ckpt/mlx/gan/opencpop/model.safetensors")
        );
        assert_eq!(
            arguments.flow_matching_checkpoint,
            PathBuf::from("../ckpt/mlx/refine/flow_matching/opencpop/model.safetensors")
        );
    }

    #[test]
    fn interface_exposes_every_refiner_and_unrestricted_file_selection() {
        assert!(
            INDEX_HTML
                .contains(r#"<input class="file-input" id="audio-file" name="audio" type="file">"#)
        );
        assert!(INDEX_HTML.contains(r#"<input type="radio" name="refiner" value="none" checked>"#));
        assert!(
            INDEX_HTML.contains(r#"<input type="radio" name="refiner" value="shallow_diffusion">"#)
        );
        assert!(
            INDEX_HTML.contains(r#"<input type="radio" name="refiner" value="flow_matching">"#)
        );
        assert!(INDEX_HTML.contains(r#"id="shallow-diffusion-parameters" hidden"#));
        assert!(INDEX_HTML.contains(r#"id="flow-matching-parameters" hidden"#));
        assert!(INDEX_HTML.contains(r#"id="slicing-parameters""#));
        assert!(INDEX_HTML.contains(
            r#"<input class="control" id="input_gain" name="input_gain" type="number" value="0" min="-12" max="12" step="1" required>"#
        ));
        assert!(!INDEX_HTML.contains("<select"));
        assert!(!INDEX_HTML.contains("shallow-diffusion-control"));
        assert!(!INDEX_HTML.contains("flow-matching-control"));
        assert!(INDEX_HTML.contains("<title>sovits-svc</title>"));
        assert!(INDEX_HTML.contains(r#"<h1 id="page-title">so-vits-svc inference</h1>"#));
        assert!(!INDEX_HTML.contains("Engine Ready"));
        assert!(!INDEX_HTML.contains("Local Voice Conversion"));
        assert!(!INDEX_HTML.contains("Babycat-supported"));
    }
}

const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="#fafafa">
  <title>sovits-svc</title>
  <style>
    :root {
      --background-100: #ffffff;
      --background-200: #fafafa;
      --gray-100: #f2f2f2;
      --gray-200: #ebebeb;
      --gray-400: #eaeaea;
      --gray-500: #c9c9c9;
      --gray-700: #8f8f8f;
      --gray-900: #4d4d4d;
      --gray-1000: #171717;
      --gray-alpha-100: #0000000d;
      --gray-alpha-200: #00000015;
      --gray-alpha-400: #00000014;
      --blue-100: #f0f7ff;
      --blue-700: #006bff;
      --blue-800: #0059ec;
      --red-100: #ffeeef;
      --red-800: #ea001d;
      --shadow-raised:
        0 1px 2px rgba(0, 0, 0, 0.04),
        0 4px 12px rgba(0, 0, 0, 0.03);
      --focus-ring: 0 0 0 2px #ffffff, 0 0 0 4px #006bff;
      font-family: Geist, "Helvetica Neue", Arial, sans-serif;
      color: var(--gray-1000);
      background: var(--background-200);
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      min-width: 320px;
      background:
        linear-gradient(var(--gray-alpha-100) 1px, transparent 1px),
        linear-gradient(90deg, var(--gray-alpha-100) 1px, transparent 1px),
        var(--background-200);
      background-size: 32px 32px;
      font-size: 14px;
      line-height: 20px;
      -webkit-font-smoothing: antialiased;
      -webkit-tap-highlight-color: rgba(0, 0, 0, 0.08);
    }

    button,
    input {
      font: inherit;
      touch-action: manipulation;
    }

    button,
    input,
    [tabindex] {
      outline: none;
    }

    button:focus-visible,
    input:focus-visible,
    [tabindex]:focus-visible {
      box-shadow: var(--focus-ring);
    }

    [hidden] {
      display: none !important;
    }

    .skip-link {
      position: fixed;
      z-index: 100;
      top: 12px;
      left: 12px;
      padding: 8px 12px;
      border-radius: 6px;
      color: #ffffff;
      background: var(--gray-1000);
      text-decoration: none;
      transform: translateY(-200%);
      transition: transform 150ms ease;
    }

    .skip-link:focus {
      transform: translateY(0);
    }

    .shell {
      width: min(1200px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0 48px;
    }

    .intro {
      margin-bottom: 20px;
    }

    h1 {
      margin: 0;
      font-size: clamp(30px, 4vw, 40px);
      font-weight: 600;
      line-height: 1.08;
      letter-spacing: -1.8px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }

    .stack {
      display: grid;
      gap: 14px;
    }

    .card {
      border: 1px solid rgba(0, 0, 0, 0.12);
      border-radius: 12px;
      background: var(--background-100);
      box-shadow: var(--shadow-raised);
      overflow: hidden;
    }

    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--gray-alpha-100);
    }

    .card-title {
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      line-height: 22px;
      letter-spacing: -0.32px;
    }

    .card-body {
      padding: 20px;
    }

    .card-body-switches,
    .card-body-refiner {
      padding: 0 20px 20px;
    }

    .drop-zone {
      display: grid;
      place-items: center;
      width: 100%;
      min-height: 160px;
      padding: 20px;
      border: 1px dashed var(--gray-500);
      border-radius: 6px;
      color: var(--gray-1000);
      background: var(--background-200);
      text-align: center;
      cursor: pointer;
      transition:
        border-color 150ms cubic-bezier(0.175, 0.885, 0.32, 1.1),
        background-color 150ms cubic-bezier(0.175, 0.885, 0.32, 1.1);
    }

    .drop-zone:hover,
    .drop-zone.is-dragging {
      border-color: var(--blue-700);
      background: var(--blue-100);
    }

    .upload-symbol {
      display: grid;
      place-items: center;
      width: 40px;
      height: 40px;
      margin: 0 auto 12px;
      border: 1px solid var(--gray-alpha-200);
      border-radius: 6px;
      background: var(--background-100);
      color: var(--gray-1000);
      font-size: 20px;
      line-height: 1;
    }

    .drop-title {
      margin: 0;
      font-size: 14px;
      font-weight: 600;
      line-height: 20px;
    }

    .file-input {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
      white-space: nowrap;
    }

    .file-summary {
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 12px;
      padding: 12px;
      border: 1px solid var(--gray-alpha-200);
      border-radius: 6px;
      background: var(--background-100);
    }

    .file-summary.is-visible {
      display: flex;
    }

    .file-name {
      overflow: hidden;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      line-height: 16px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .file-size {
      flex: none;
      color: var(--gray-900);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      line-height: 16px;
    }

    .field-error {
      margin: 8px 0 0;
      color: var(--red-800);
      font-size: 12px;
      line-height: 16px;
    }

    .field-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    .field-grid-two {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .field {
      display: grid;
      gap: 8px;
      min-width: 0;
    }

    .field-wide {
      grid-column: 1 / -1;
    }

    .field label,
    .field-label {
      font-size: 13px;
      font-weight: 500;
      line-height: 16px;
    }

    .field-hint {
      color: var(--gray-900);
      font-size: 12px;
      font-weight: 400;
      line-height: 16px;
    }

    .type-selector {
      min-width: 0;
      margin: 16px 0 0;
      padding: 0;
      border: 0;
    }

    .type-selector legend {
      margin-bottom: 8px;
      padding: 0;
      font-size: 13px;
      font-weight: 500;
      line-height: 16px;
    }

    .type-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }

    .type-option {
      position: relative;
      display: block;
      height: 100%;
      min-width: 0;
      cursor: pointer;
    }

    .type-option input {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0 0 0 0);
      white-space: nowrap;
    }

    .type-option-body {
      display: grid;
      place-items: center;
      height: 100%;
      min-height: 44px;
      padding: 10px;
      border: 1px solid var(--gray-alpha-200);
      border-radius: 8px;
      background: var(--background-100);
      text-align: center;
      transition:
        border-color 150ms ease,
        box-shadow 150ms ease,
        background-color 150ms ease;
    }

    .type-option:hover .type-option-body {
      border-color: var(--gray-500);
      background: var(--background-200);
    }

    .type-option input:checked + .type-option-body {
      border-color: var(--gray-1000);
      background: var(--background-100);
      box-shadow:
        inset 0 0 0 1px var(--gray-1000),
        0 1px 2px rgba(0, 0, 0, 0.04);
    }

    .type-option input:focus-visible + .type-option-body {
      box-shadow: var(--focus-ring);
    }

    .type-option-title {
      font-size: 13px;
      font-weight: 600;
      line-height: 18px;
    }

    .control {
      width: 100%;
      height: 40px;
      padding: 0 12px;
      border: 1px solid var(--gray-alpha-200);
      border-radius: 6px;
      color: var(--gray-1000);
      background: var(--background-100);
    }

    .control:hover {
      border-color: var(--gray-500);
    }

    .pitch-control {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
    }

    .pitch-control .control {
      position: relative;
      border-radius: 0;
    }

    .pitch-button {
      height: 40px;
      padding: 0 12px;
      border: 1px solid var(--gray-alpha-200);
      color: var(--gray-1000);
      background: var(--background-100);
      cursor: pointer;
    }

    .pitch-button:first-child {
      border-right: 0;
      border-radius: 6px 0 0 6px;
    }

    .pitch-button:last-child {
      border-left: 0;
      border-radius: 0 6px 6px 0;
    }

    .pitch-button:hover {
      background: var(--gray-100);
    }

    .pitch-button[aria-pressed="true"] {
      color: #ffffff;
      background: var(--gray-1000);
    }

    .switch-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 48px;
      padding: 8px 0;
      border-bottom: 1px solid var(--gray-alpha-100);
      cursor: pointer;
    }

    .switch-row:last-child {
      border-bottom: 0;
    }

    .switch-row:hover .switch-title {
      color: var(--blue-800);
    }

    .switch-copy {
      display: grid;
      gap: 2px;
    }

    .switch-title {
      font-size: 13px;
      font-weight: 500;
      line-height: 16px;
    }

    .switch {
      position: relative;
      flex: none;
      width: 36px;
      height: 20px;
    }

    .switch input {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      margin: 0;
      opacity: 0;
      cursor: pointer;
    }

    .switch-track {
      position: absolute;
      inset: 0;
      border-radius: 9999px;
      background: var(--gray-500);
      pointer-events: none;
      transition: background-color 150ms cubic-bezier(0.175, 0.885, 0.32, 1.1);
    }

    .switch-track::after {
      content: "";
      position: absolute;
      top: 2px;
      left: 2px;
      width: 16px;
      height: 16px;
      border-radius: 9999px;
      background: #ffffff;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
      transition: transform 150ms cubic-bezier(0.175, 0.885, 0.32, 1.1);
    }

    .switch input:checked + .switch-track {
      background: var(--gray-1000);
    }

    .switch input:checked + .switch-track::after {
      transform: translateX(16px);
    }

    .switch input:focus-visible + .switch-track {
      box-shadow: var(--focus-ring);
    }

    .parameter-panel {
      display: grid;
      gap: 14px;
    }

    .parameter-panel-fields {
      padding-top: 20px;
    }

    .action-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-top: 14px;
      padding: 14px;
      border: 1px solid var(--gray-alpha-200);
      border-radius: 12px;
      background: var(--background-100);
      box-shadow: var(--shadow-raised);
    }

    .action-copy {
      color: var(--gray-900);
      font-size: 13px;
      line-height: 18px;
    }

    .primary-button,
    .secondary-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      height: 40px;
      padding: 0 14px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 500;
      line-height: 20px;
      text-decoration: none;
      cursor: pointer;
    }

    .primary-button {
      border: 1px solid var(--gray-1000);
      color: #ffffff;
      background: var(--gray-1000);
    }

    .primary-button:hover {
      background: var(--gray-900);
    }

    .primary-button:disabled {
      opacity: 0.72;
      cursor: not-allowed;
    }

    .button-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255, 255, 255, 0.35);
      border-top-color: #ffffff;
      border-radius: 9999px;
      animation: spin 700ms linear infinite;
    }

    @keyframes spin {
      to {
        transform: rotate(360deg);
      }
    }

    .secondary-button {
      border: 1px solid var(--gray-alpha-200);
      color: var(--gray-1000);
      background: var(--background-100);
    }

    .secondary-button:hover {
      background: var(--gray-100);
    }

    .message {
      display: none;
      margin-top: 14px;
      padding: 12px 16px;
      border: 1px solid;
      border-radius: 6px;
      font-size: 13px;
      line-height: 18px;
    }

    .message.is-visible {
      display: block;
    }

    .message-error {
      border-color: #ffb1b3;
      color: var(--red-800);
      background: var(--red-100);
    }

    .result {
      display: none;
      margin-top: 14px;
    }

    .result.is-visible {
      display: block;
    }

    .result-content {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      align-items: center;
    }

    .result-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
      margin-top: 8px;
      color: var(--gray-900);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      line-height: 16px;
    }

    audio {
      width: 100%;
      height: 40px;
      margin-top: 14px;
    }

    @media (max-width: 960px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .field-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 600px) {
      input {
        font-size: 16px;
      }

      .shell {
        width: min(100% - 24px, 1200px);
        padding-top: 16px;
      }

      .intro {
        margin-bottom: 16px;
      }

      h1 {
        letter-spacing: -1.28px;
      }

      .card-header {
        padding: 12px 16px;
      }

      .card-body {
        padding: 16px;
      }

      .card-body-switches,
      .card-body-refiner {
        padding: 0 16px 16px;
      }

      .parameter-panel-fields {
        padding-top: 16px;
      }

      .field-grid {
        grid-template-columns: 1fr;
      }

      .type-grid {
        grid-template-columns: 1fr;
      }

      .control,
      .pitch-button,
      .primary-button,
      .secondary-button {
        min-height: 44px;
      }

      .action-bar,
      .result-content {
        grid-template-columns: 1fr;
        align-items: stretch;
      }

      .action-bar {
        display: grid;
      }

      .primary-button,
      .secondary-button {
        width: 100%;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        scroll-behavior: auto !important;
        transition-duration: 0ms !important;
        animation-duration: 0ms !important;
      }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to Content</a>
  <div class="shell">
    <main id="main-content">
      <section class="intro" aria-labelledby="page-title">
        <h1 id="page-title">so-vits-svc inference</h1>
      </section>

      <form id="inference-form">
        <div class="layout">
          <div class="stack">
            <section class="card" aria-labelledby="input-title">
              <div class="card-header">
                <h2 class="card-title" id="input-title">Input Audio</h2>
              </div>
              <div class="card-body">
                <button class="drop-zone" id="drop-zone" type="button" aria-controls="audio-file">
                  <div>
                    <span class="upload-symbol" aria-hidden="true">↑</span>
                    <p class="drop-title">Drop an audio file here</p>
                  </div>
                </button>
                <input class="file-input" id="audio-file" name="audio" type="file">
                <div class="file-summary" id="file-summary" role="status" aria-live="polite">
                  <span class="file-name" id="file-name"></span>
                  <span class="file-size" id="file-size"></span>
                </div>
                <p class="field-error" id="file-error" role="alert" hidden>Choose an audio file before running inference.</p>
              </div>
            </section>

            <section class="card" aria-labelledby="behavior-title">
              <div class="card-header">
                <h2 class="card-title" id="behavior-title">Pipeline</h2>
              </div>
              <div class="card-body card-body-switches">
                <label class="switch-row" for="predict_f0">
                  <span class="switch-copy">
                    <span class="switch-title">Automatic F0</span>
                  </span>
                  <span class="switch">
                    <input id="predict_f0" name="predict_f0" type="checkbox" aria-label="Enable automatic F0">
                    <span class="switch-track" aria-hidden="true"></span>
                  </span>
                </label>
                <label class="switch-row" for="slicing">
                  <span class="switch-copy">
                    <span class="switch-title">Silence Slicing</span>
                  </span>
                  <span class="switch">
                    <input id="slicing" name="slicing" type="checkbox" checked aria-label="Enable silence slicing">
                    <span class="switch-track" aria-hidden="true"></span>
                  </span>
                </label>
                <fieldset class="type-selector">
                  <legend>Refiner Type</legend>
                  <div class="type-grid">
                    <label class="type-option">
                      <input type="radio" name="refiner" value="none" checked>
                      <span class="type-option-body">
                        <span class="type-option-title">GAN Only</span>
                      </span>
                    </label>
                    <label class="type-option">
                      <input type="radio" name="refiner" value="shallow_diffusion">
                      <span class="type-option-body">
                        <span class="type-option-title">Shallow Diffusion</span>
                      </span>
                    </label>
                    <label class="type-option">
                      <input type="radio" name="refiner" value="flow_matching">
                      <span class="type-option-body">
                        <span class="type-option-title">Flow Matching</span>
                      </span>
                    </label>
                  </div>
                </fieldset>
              </div>
            </section>
          </div>

          <div class="stack">
            <section class="card" aria-labelledby="inference-title">
              <div class="card-header">
                <h2 class="card-title" id="inference-title">Inference Parameters</h2>
              </div>
              <div class="card-body">
                <div class="field-grid field-grid-two">
                  <div class="field">
                    <label for="input_gain">Input Gain</label>
                    <input class="control" id="input_gain" name="input_gain" type="number" value="0" min="-12" max="12" step="1" required>
                    <span class="field-hint">dB</span>
                  </div>
                  <div class="field">
                    <label for="pitch_shift">Pitch Shift</label>
                    <div class="pitch-control">
                      <button class="pitch-button" type="button" data-pitch-value="-12" aria-pressed="false">−12</button>
                      <input class="control" id="pitch_shift" name="pitch_shift" type="number" value="0" min="-48" max="48" step="1" required>
                      <button class="pitch-button" type="button" data-pitch-value="12" aria-pressed="false">+12</button>
                    </div>
                    <span class="field-hint">Semitones</span>
                  </div>
                  <div class="field">
                    <label for="noise_scale">GAN Noise Scale</label>
                    <input class="control" id="noise_scale" name="noise_scale" type="number" value="0.4" min="0" step="0.01" required>
                    <span class="field-hint">Non-negative</span>
                  </div>
                  <div class="field">
                    <label for="loudness_envelope_adjustment">Loudness Envelope</label>
                    <input class="control" id="loudness_envelope_adjustment" name="loudness_envelope_adjustment" type="number" value="0" min="0" max="1" step="0.05" required>
                    <span class="field-hint">0–1 strength</span>
                  </div>
                </div>
              </div>
            </section>

            <section class="card" id="refiner-parameters" aria-labelledby="refiner-parameters-title" hidden>
              <div class="card-header">
                <h2 class="card-title" id="refiner-parameters-title">Refiner Parameters</h2>
              </div>
              <div class="card-body card-body-refiner">
                <div class="parameter-panel" id="shallow-diffusion-parameters" hidden>
                  <label class="switch-row" for="second_encoding">
                    <span class="switch-copy">
                      <span class="switch-title">Second Encoding</span>
                    </span>
                    <span class="switch">
                      <input id="second_encoding" name="second_encoding" type="checkbox" aria-label="Enable second encoding">
                      <span class="switch-track" aria-hidden="true"></span>
                    </span>
                  </label>
                  <div class="field-grid">
                    <div class="field">
                    <label for="diffusion_steps">Diffusion Steps</label>
                    <input class="control" id="diffusion_steps" name="diffusion_steps" type="number" value="100" min="2" max="1000" step="1" required>
                    <span class="field-hint">Schedule depth</span>
                    </div>
                    <div class="field">
                    <label for="diffusion_speedup">Diffusion Speedup</label>
                    <input class="control" id="diffusion_speedup" name="diffusion_speedup" type="number" value="10" min="1" max="50" step="1" required>
                    <span class="field-hint">Steps ÷ speedup ≥ 2</span>
                    </div>
                  </div>
                </div>
                <div class="parameter-panel parameter-panel-fields" id="flow-matching-parameters" hidden>
                  <div class="field-grid">
                  <div class="field">
                    <label for="flow_matching_steps">Flow Matching Steps</label>
                    <input class="control" id="flow_matching_steps" name="flow_matching_steps" type="number" value="50" min="1" step="1" required>
                    <span class="field-hint">Euler ODE steps</span>
                  </div>
                  </div>
                </div>
              </div>
            </section>

            <section class="card" id="slicing-parameters" aria-labelledby="slicing-title">
              <div class="card-header">
                <h2 class="card-title" id="slicing-title">Slicing Parameters</h2>
              </div>
              <div class="card-body">
                <div class="field-grid">
                  <div class="field">
                    <label for="threshold_db">Silence Threshold</label>
                    <input class="control slice-control" id="threshold_db" name="threshold_db" type="number" value="-40" max="0" step="0.1" required>
                    <span class="field-hint">dB</span>
                  </div>
                  <div class="field">
                    <label for="padding_seconds">Context Padding</label>
                    <input class="control slice-control" id="padding_seconds" name="padding_seconds" type="number" value="0.5" min="0" step="0.05" required>
                    <span class="field-hint">Seconds per side</span>
                  </div>
                  <div class="field">
                    <label for="clip_seconds">Maximum Chunk</label>
                    <input class="control slice-control" id="clip_seconds" name="clip_seconds" type="number" value="0" min="0" step="0.1" required>
                    <span class="field-hint">0 disables splitting</span>
                  </div>
                  <div class="field">
                    <label for="crossfade_seconds">Chunk Overlap</label>
                    <input class="control slice-control" id="crossfade_seconds" name="crossfade_seconds" type="number" value="0" min="0" step="0.05" required>
                    <span class="field-hint">Seconds</span>
                  </div>
                  <div class="field">
                    <label for="crossfade_ratio">Crossfade Ratio</label>
                    <input class="control slice-control" id="crossfade_ratio" name="crossfade_ratio" type="number" value="0.75" min="0" max="1" step="0.05" required>
                    <span class="field-hint">0–1 overlap fraction</span>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </div>

        <div class="action-bar">
          <span class="action-copy" id="action-copy" aria-live="polite">Choose an audio file, then run inference.</span>
          <button class="primary-button" id="run-button" type="submit">
            <span class="button-spinner" id="button-spinner" aria-hidden="true" hidden></span>
            <span>Run Inference</span>
          </button>
        </div>

        <div class="message message-error" id="error-message" role="alert" aria-live="assertive" tabindex="-1"></div>

        <section class="card result" id="result" aria-labelledby="result-title" tabindex="-1">
          <div class="card-body">
            <div class="result-content">
              <div>
                <h2 class="card-title" id="result-title">Converted Audio</h2>
                <div class="result-meta">
                  <span id="result-duration"></span>
                  <span id="result-time"></span>
                  <span id="result-size"></span>
                </div>
              </div>
              <a class="secondary-button" id="download-button" href="#" download>Download WAV</a>
            </div>
            <audio id="output-audio" controls></audio>
          </div>
        </section>
      </form>
    </main>
  </div>

  <script>
    const byId = (id) => document.getElementById(id);
    const form = byId("inference-form");
    const dropZone = byId("drop-zone");
    const fileInput = byId("audio-file");
    const fileSummary = byId("file-summary");
    const fileError = byId("file-error");
    const runButton = byId("run-button");
    const buttonSpinner = byId("button-spinner");
    const actionCopy = byId("action-copy");
    const errorMessage = byId("error-message");
    const slicing = byId("slicing");
    const slicingParameters = byId("slicing-parameters");
    const refinerInputs = Array.from(document.querySelectorAll('input[name="refiner"]'));
    const refinerParameters = byId("refiner-parameters");
    const shallowDiffusionParameters = byId("shallow-diffusion-parameters");
    const flowMatchingParameters = byId("flow-matching-parameters");
    const pitchShift = byId("pitch_shift");
    const diffusionSteps = byId("diffusion_steps");
    const diffusionSpeedup = byId("diffusion_speedup");
    const result = byId("result");
    const outputAudio = byId("output-audio");
    const downloadButton = byId("download-button");
    let selectedFile = null;
    let outputUrl = null;

    function formatBytes(bytes) {
      if (bytes < 1024) return bytes + " B";
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
      return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function outputName(filename) {
      const basename = filename.split(/[\\/]/).pop() || "audio";
      const dot = basename.lastIndexOf(".");
      const stem = dot > 0 ? basename.slice(0, dot) : basename;
      return (stem || "audio") + "-converted.wav";
    }

    function selectFile(file) {
      if (!file) return;
      selectedFile = file;
      byId("file-name").textContent = file.name;
      byId("file-size").textContent = formatBytes(file.size);
      fileSummary.classList.add("is-visible");
      actionCopy.textContent = "Ready to run local MLX inference.";
      fileError.hidden = true;
      hideError();
    }

    function showError(message, focusTarget = errorMessage) {
      errorMessage.textContent = message;
      errorMessage.classList.add("is-visible");
      focusTarget.focus();
    }

    function hideError() {
      errorMessage.textContent = "";
      errorMessage.classList.remove("is-visible");
    }

    function updateSlicingState() {
      slicingParameters.hidden = !slicing.checked;
      slicingParameters.querySelectorAll("input").forEach((control) => {
        control.disabled = !slicing.checked;
      });
    }

    function selectedRefiner() {
      return refinerInputs.find((input) => input.checked).value;
    }

    function updateRefinerState() {
      const refiner = selectedRefiner();
      const shallowDiffusion = refiner === "shallow_diffusion";
      const flowMatching = refiner === "flow_matching";
      refinerParameters.hidden = refiner === "none";
      shallowDiffusionParameters.hidden = !shallowDiffusion;
      flowMatchingParameters.hidden = !flowMatching;
      shallowDiffusionParameters.querySelectorAll("input").forEach((control) => {
        control.disabled = !shallowDiffusion;
      });
      flowMatchingParameters.querySelectorAll("input").forEach((control) => {
        control.disabled = !flowMatching;
      });
    }

    function updateSpeedupLimit() {
      const steps = Number(diffusionSteps.value);
      const maximum = Math.max(1, Math.floor(steps / 2));
      diffusionSpeedup.max = String(maximum);
      if (Number(diffusionSpeedup.value) > maximum) {
        diffusionSpeedup.value = String(maximum);
      }
    }

    function updatePitchButtons() {
      document.querySelectorAll("[data-pitch-value]").forEach((button) => {
        button.setAttribute(
          "aria-pressed",
          String(Number(pitchShift.value) === Number(button.dataset.pitchValue))
        );
      });
    }

    async function waitForMinimumLoading(startedAt) {
      const remaining = 400 - (performance.now() - startedAt);
      if (remaining > 0) {
        await new Promise((resolve) => setTimeout(resolve, remaining));
      }
    }

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-dragging");
      });
    });
    dropZone.addEventListener("drop", (event) => {
      selectFile(event.dataTransfer.files[0]);
    });

    slicing.addEventListener("change", updateSlicingState);
    refinerInputs.forEach((input) => input.addEventListener("change", updateRefinerState));
    pitchShift.addEventListener("input", updatePitchButtons);
    document.querySelectorAll("[data-pitch-value]").forEach((button) => {
      button.addEventListener("click", () => {
        pitchShift.value = button.dataset.pitchValue;
        updatePitchButtons();
      });
    });
    diffusionSteps.addEventListener("input", updateSpeedupLimit);
    updateSlicingState();
    updateRefinerState();
    updatePitchButtons();
    updateSpeedupLimit();

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!selectedFile) {
        fileError.hidden = false;
        dropZone.focus();
        return;
      }
      if (!form.reportValidity()) return;

      const loadingStartedAt = performance.now();
      const refiner = selectedRefiner();
      hideError();
      result.classList.remove("is-visible");
      runButton.disabled = true;
      runButton.setAttribute("aria-busy", "true");
      buttonSpinner.hidden = false;
      actionCopy.textContent = "Running inference locally…";

      const body = new FormData();
      body.append("audio", selectedFile, selectedFile.name);
      [
        "input_gain",
        "pitch_shift",
        "noise_scale",
        "loudness_envelope_adjustment"
      ].forEach((name) => body.append(name, byId(name).value));
      body.append("predict_f0", String(byId("predict_f0").checked));
      body.append("refiner", refiner);
      body.append("slicing", String(slicing.checked));
      if (refiner === "shallow_diffusion") {
        body.append("diffusion_steps", diffusionSteps.value);
        body.append("diffusion_speedup", diffusionSpeedup.value);
        body.append("second_encoding", String(byId("second_encoding").checked));
      } else if (refiner === "flow_matching") {
        body.append("flow_matching_steps", byId("flow_matching_steps").value);
      }
      if (slicing.checked) {
        [
          "threshold_db",
          "padding_seconds",
          "clip_seconds",
          "crossfade_seconds",
          "crossfade_ratio"
        ].forEach((name) => body.append(name, byId(name).value));
      }

      try {
        const response = await fetch("/api/infer", { method: "POST", body });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const blob = await response.blob();
        await waitForMinimumLoading(loadingStartedAt);
        if (outputUrl) URL.revokeObjectURL(outputUrl);
        outputUrl = URL.createObjectURL(blob);
        const derivedOutputName = outputName(selectedFile.name);
        const sampleCount = Number(response.headers.get("x-output-samples") || 0);
        const inferenceMilliseconds = Number(response.headers.get("x-inference-ms") || 0);
        outputAudio.src = outputUrl;
        downloadButton.href = outputUrl;
        downloadButton.download = derivedOutputName;
        byId("result-duration").textContent =
          sampleCount > 0 ? (sampleCount / 44100).toFixed(2) + " s audio" : "";
        byId("result-time").textContent =
          inferenceMilliseconds > 0 ? (inferenceMilliseconds / 1000).toFixed(2) + " s inference" : "";
        byId("result-size").textContent = formatBytes(blob.size);
        result.classList.add("is-visible");
        result.focus();
        actionCopy.textContent = "Inference complete. Preview or download the result.";
      } catch (error) {
        await waitForMinimumLoading(loadingStartedAt);
        const message = error.message || "Inference could not complete.";
        showError(`${message} Review the audio format and parameters, then try again.`);
        actionCopy.textContent = "Something went wrong. Review the guidance below.";
      } finally {
        runButton.disabled = false;
        runButton.removeAttribute("aria-busy");
        buttonSpinner.hidden = true;
      }
    });
  </script>
</body>
</html>
"##;
