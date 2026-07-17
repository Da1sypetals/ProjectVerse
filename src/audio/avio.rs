use std::ffi::{CString, c_int, c_void};
use std::marker::PhantomData;
use std::ptr;

use anyhow::{Context, Result, ensure};
use ffmpeg_the_third as ffmpeg;
use ffmpeg_the_third::ffi;

const AVIO_BUFFER_SIZE: usize = 32 * 1024;

struct MemoryCursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

unsafe extern "C" fn read_packet(
    opaque: *mut c_void,
    buffer: *mut u8,
    buffer_size: c_int,
) -> c_int {
    if opaque.is_null() || buffer.is_null() || buffer_size <= 0 {
        return ffi::AVERROR(libc::EINVAL);
    }

    let cursor = unsafe { &mut *(opaque as *mut MemoryCursor<'_>) };
    if cursor.position == cursor.bytes.len() {
        return ffi::AVERROR_EOF;
    }

    let requested = buffer_size as usize;
    let available = cursor.bytes.len() - cursor.position;
    let count = requested.min(available);
    unsafe {
        ptr::copy_nonoverlapping(cursor.bytes.as_ptr().add(cursor.position), buffer, count);
    }
    cursor.position += count;
    count as c_int
}

unsafe extern "C" fn seek(opaque: *mut c_void, offset: i64, whence: c_int) -> i64 {
    if opaque.is_null() {
        return i64::from(ffi::AVERROR(libc::EINVAL));
    }

    let cursor = unsafe { &mut *(opaque as *mut MemoryCursor<'_>) };
    if whence & ffi::AVSEEK_SIZE as c_int != 0 {
        return match i64::try_from(cursor.bytes.len()) {
            Ok(length) => length,
            Err(_) => i64::from(ffi::AVERROR(libc::EOVERFLOW)),
        };
    }

    let base = match whence & !(ffi::AVSEEK_FORCE as c_int) {
        libc::SEEK_SET => 0_i64,
        libc::SEEK_CUR => match i64::try_from(cursor.position) {
            Ok(position) => position,
            Err(_) => return i64::from(ffi::AVERROR(libc::EOVERFLOW)),
        },
        libc::SEEK_END => match i64::try_from(cursor.bytes.len()) {
            Ok(length) => length,
            Err(_) => return i64::from(ffi::AVERROR(libc::EOVERFLOW)),
        },
        _ => return i64::from(ffi::AVERROR(libc::EINVAL)),
    };
    let Some(position) = base.checked_add(offset) else {
        return i64::from(ffi::AVERROR(libc::EOVERFLOW));
    };
    let Ok(position_usize) = usize::try_from(position) else {
        return i64::from(ffi::AVERROR(libc::EINVAL));
    };
    if position_usize > cursor.bytes.len() {
        return i64::from(ffi::AVERROR(libc::EINVAL));
    }
    cursor.position = position_usize;
    position
}

struct AvioContext<'a> {
    context: *mut ffi::AVIOContext,
    cursor: *mut MemoryCursor<'a>,
    _lifetime: PhantomData<&'a [u8]>,
}

impl<'a> AvioContext<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        ensure!(!bytes.is_empty(), "uploaded audio is empty");
        let buffer_size = c_int::try_from(AVIO_BUFFER_SIZE).expect("AVIO buffer size fits c_int");
        let buffer = unsafe { ffi::av_malloc(AVIO_BUFFER_SIZE) as *mut u8 };
        ensure!(!buffer.is_null(), "failed to allocate FFmpeg AVIO buffer");

        let cursor = Box::into_raw(Box::new(MemoryCursor { bytes, position: 0 }));
        let context = unsafe {
            ffi::avio_alloc_context(
                buffer,
                buffer_size,
                0,
                cursor.cast(),
                Some(read_packet),
                None,
                Some(seek),
            )
        };
        if context.is_null() {
            unsafe {
                ffi::av_free(buffer.cast());
                drop(Box::from_raw(cursor));
            }
            anyhow::bail!("failed to allocate FFmpeg AVIO context");
        }

        Ok(Self {
            context,
            cursor,
            _lifetime: PhantomData,
        })
    }
}

impl Drop for AvioContext<'_> {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                let buffer = (*self.context).buffer;
                if !buffer.is_null() {
                    ffi::av_free(buffer.cast());
                    (*self.context).buffer = ptr::null_mut();
                }
                ffi::avio_context_free(&mut self.context);
            }
            if !self.cursor.is_null() {
                drop(Box::from_raw(self.cursor));
                self.cursor = ptr::null_mut();
            }
        }
    }
}

pub(super) struct MemoryInput<'a> {
    input: Option<ffmpeg::format::context::Input>,
    avio: Option<AvioContext<'a>>,
}

impl<'a> MemoryInput<'a> {
    pub(super) fn open(bytes: &'a [u8], file_extension: &str, mime_type: &str) -> Result<Self> {
        let avio = AvioContext::new(bytes)?;
        let filename = if file_extension.is_empty() {
            "upload".to_owned()
        } else {
            format!("upload.{}", file_extension.trim_start_matches('.'))
        };
        let filename = CString::new(filename).context("uploaded audio extension contains NUL")?;

        let mut raw_input = unsafe { ffi::avformat_alloc_context() };
        ensure!(
            !raw_input.is_null(),
            "failed to allocate FFmpeg format context"
        );
        unsafe {
            (*raw_input).pb = avio.context;
            (*raw_input).flags |= ffi::AVFMT_FLAG_CUSTOM_IO as c_int;
        }

        let open_result = unsafe {
            ffi::avformat_open_input(
                &mut raw_input,
                filename.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if open_result < 0 {
            if !raw_input.is_null() {
                unsafe { ffi::avformat_close_input(&mut raw_input) };
            }
            return Err(ffmpeg::Error::from(open_result)).with_context(|| {
                format!(
                    "failed to identify uploaded audio (extension={file_extension:?}, mime={mime_type:?})"
                )
            });
        }

        let stream_info_result =
            unsafe { ffi::avformat_find_stream_info(raw_input, ptr::null_mut()) };
        if stream_info_result < 0 {
            unsafe { ffi::avformat_close_input(&mut raw_input) };
            return Err(ffmpeg::Error::from(stream_info_result))
                .context("failed to read uploaded audio stream information");
        }

        let input = unsafe { ffmpeg::format::context::Input::wrap(raw_input) };
        Ok(Self {
            input: Some(input),
            avio: Some(avio),
        })
    }

    pub(super) fn input_mut(&mut self) -> &mut ffmpeg::format::context::Input {
        self.input
            .as_mut()
            .expect("memory input context is available until drop")
    }
}

impl Drop for MemoryInput<'_> {
    fn drop(&mut self) {
        drop(self.input.take());
        drop(self.avio.take());
    }
}
