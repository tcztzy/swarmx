use std::fs;
use std::io::{self, BufRead, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use typst::diag::{FileError, FileResult, SourceDiagnostic};
use typst::foundations::{Bytes, Datetime, Duration};
use typst::introspection::PagedPosition;
use typst::layout::Point;
use typst::syntax::{FileId, RootedPath, Source, VirtualPath, VirtualRoot};
use typst::text::{Font, FontBook};
use typst::utils::LazyHash;
use typst::{Library, LibraryExt, World};
use typst_ide::{IdeWorld, Jump, jump_from_click};
use typst_kit::datetime::Time;
use typst_kit::files::{FileLoader, FileStore};
use typst_kit::fonts::{self, FontStore};
use typst_layout::PagedDocument;
use typst_pdf::PdfOptions;

const ENGINE: &str = "typst";

#[derive(Debug)]
struct Args {
    root: PathBuf,
    input: PathBuf,
    output_directory: PathBuf,
    max_pdf_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum Request {
    Resolve {
        id: String,
        #[serde(rename = "pdfRevision")]
        pdf_revision: String,
        page: usize,
        x: f64,
        y: f64,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SourceTarget {
    relative_path: String,
    source_revision: String,
    offset: usize,
}

#[derive(Debug)]
enum Input {
    Request(Request),
    Invalid(String),
    Closed,
}

struct ProjectFiles {
    root: PathBuf,
}

impl ProjectFiles {
    fn resolve(&self, id: FileId) -> FileResult<PathBuf> {
        if !matches!(id.root(), VirtualRoot::Project) {
            return Err(FileError::AccessDenied);
        }
        id.vpath().realize(&self.root).map_err(FileError::Realize)
    }
}

impl FileLoader for ProjectFiles {
    fn load(&self, id: FileId) -> FileResult<Bytes> {
        let unresolved = self.resolve(id)?;
        let canonical = fs::canonicalize(&unresolved)
            .map_err(|error| FileError::from_io(error, &unresolved))?;
        if !canonical.starts_with(&self.root) {
            return Err(FileError::AccessDenied);
        }
        let metadata =
            fs::metadata(&canonical).map_err(|error| FileError::from_io(error, &canonical))?;
        if !metadata.is_file() {
            return Err(FileError::IsDirectory);
        }
        fs::read(&canonical)
            .map(Bytes::new)
            .map_err(|error| FileError::from_io(error, &canonical))
    }
}

struct BridgeWorld {
    library: LazyHash<Library>,
    fonts: Arc<FontStore>,
    files: FileStore<ProjectFiles>,
    main: FileId,
    time: Time,
}

impl BridgeWorld {
    fn new(root: PathBuf, input: &Path, fonts: Arc<FontStore>) -> Result<Self, String> {
        let vpath = VirtualPath::virtualize(&root, input).map_err(|error| error.to_string())?;
        let main = RootedPath::new(VirtualRoot::Project, vpath).intern();
        Ok(Self {
            library: LazyHash::new(Library::builder().build()),
            fonts,
            files: FileStore::new(ProjectFiles { root }),
            main,
            time: Time::system(),
        })
    }

    fn dependencies(&mut self) -> Vec<PathBuf> {
        let (loader, ids) = self.files.dependencies();
        ids.filter_map(|id| loader.resolve(id).ok()).collect()
    }
}

impl World for BridgeWorld {
    fn library(&self) -> &LazyHash<Library> {
        &self.library
    }

    fn book(&self) -> &LazyHash<FontBook> {
        self.fonts.book()
    }

    fn main(&self) -> FileId {
        self.main
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        self.files.source(id)
    }

    fn file(&self, id: FileId) -> FileResult<Bytes> {
        self.files.file(id)
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.fonts.font(index)
    }

    fn today(&self, offset: Option<Duration>) -> Option<Datetime> {
        self.time.today(offset)
    }
}

impl IdeWorld for BridgeWorld {
    fn upcast(&self) -> &dyn World {
        self
    }
}

struct Snapshot {
    world: Arc<BridgeWorld>,
    document: PagedDocument,
    pdf_revision: String,
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes.as_ref()))
}

fn diagnostics(items: &[SourceDiagnostic]) -> Vec<String> {
    items
        .iter()
        .take(100)
        .map(|item| item.message.to_string().chars().take(4_096).collect())
        .collect()
}

fn emit(value: serde_json::Value) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()
}

fn compile(
    args: &Args,
    fonts: Arc<FontStore>,
    previous: Option<Snapshot>,
) -> (Option<Snapshot>, Vec<PathBuf>) {
    let mut world = match BridgeWorld::new(args.root.clone(), &args.input, fonts) {
        Ok(world) => world,
        Err(error) => {
            let _ = emit(json!({
                "type": "compile-error",
                "engine": ENGINE,
                "diagnostics": [error],
            }));
            return (previous, vec![args.input.clone()]);
        }
    };
    let compiled = typst::compile::<PagedDocument>(&world);
    let warnings = diagnostics(&compiled.warnings);
    let output = match compiled.output {
        Ok(document) => match typst_pdf::pdf(&document, &PdfOptions::default()) {
            Ok(pdf) => Some((document, pdf)),
            Err(errors) => {
                let _ = emit(json!({
                    "type": "compile-error",
                    "engine": ENGINE,
                    "diagnostics": diagnostics(&errors),
                }));
                None
            }
        },
        Err(errors) => {
            let _ = emit(json!({
                "type": "compile-error",
                "engine": ENGINE,
                "diagnostics": diagnostics(&errors),
            }));
            None
        }
    };
    let dependencies = world.dependencies();
    let Some((document, pdf)) = output else {
        return (previous, dependencies);
    };
    if pdf.len() > args.max_pdf_bytes {
        let _ = emit(json!({
            "type": "compile-error",
            "engine": ENGINE,
            "diagnostics": ["Compiled PDF exceeds the configured preview limit."],
        }));
        return (previous, dependencies);
    }
    let pdf_revision = digest(&pdf);
    let source_revision = world
        .source(world.main())
        .map(|source| digest(source.text().as_bytes()))
        .unwrap_or_else(|_| digest([]));
    let filename = format!("preview-{}.pdf", &pdf_revision[7..]);
    let output_path = args.output_directory.join(filename);
    let temporary_path = args.output_directory.join("preview.tmp");
    let written =
        fs::write(&temporary_path, &pdf).and_then(|_| fs::rename(&temporary_path, &output_path));
    if let Err(error) = written {
        let _ = emit(json!({
            "type": "compile-error",
            "engine": ENGINE,
            "diagnostics": [format!("Unable to publish PDF preview: {error}")],
        }));
        return (previous, dependencies);
    }
    let _ = emit(json!({
        "type": "compiled",
        "engine": ENGINE,
        "pdfFile": output_path,
        "pdfRevision": pdf_revision,
        "sourceRevision": source_revision,
        "pdfSize": pdf.len(),
        "compiledAt": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
        "diagnostics": warnings,
    }));
    (
        Some(Snapshot {
            world: Arc::new(world),
            document,
            pdf_revision,
        }),
        dependencies,
    )
}

fn resolve_click(snapshot: Option<&Snapshot>, request: Request) {
    let Request::Resolve {
        id,
        pdf_revision,
        page,
        x,
        y,
    } = request;
    let target = snapshot
        .filter(|snapshot| snapshot.pdf_revision == pdf_revision)
        .filter(|_| {
            page > 0
                && x.is_finite()
                && y.is_finite()
                && (0.0..=1.0).contains(&x)
                && (0.0..=1.0).contains(&y)
        })
        .and_then(|snapshot| {
            let page_frame = snapshot.document.pages().get(page - 1)?;
            let position = PagedPosition {
                page: NonZeroUsize::new(page)?,
                point: Point::new(page_frame.frame.size().x * x, page_frame.frame.size().y * y),
            };
            match jump_from_click(snapshot.world.as_ref(), &snapshot.document, &position)? {
                Jump::File(file_id, byte_offset)
                    if matches!(file_id.root(), VirtualRoot::Project) =>
                {
                    let source = snapshot.world.source(file_id).ok()?;
                    let prefix = source.text().get(..byte_offset)?;
                    Some(SourceTarget {
                        relative_path: file_id.vpath().get_without_slash().to_owned(),
                        source_revision: digest(source.text().as_bytes()),
                        offset: prefix.encode_utf16().count(),
                    })
                }
                _ => None,
            }
        });
    let _ = emit(json!({
        "type": "resolved",
        "engine": ENGINE,
        "id": id,
        "pdfRevision": pdf_revision,
        "target": target,
    }));
}

#[derive(Clone, Eq, PartialEq)]
struct FileState {
    path: PathBuf,
    size: Option<u64>,
    modified: Option<u128>,
}

fn file_states(paths: &[PathBuf]) -> Vec<FileState> {
    paths
        .iter()
        .map(|path| {
            let metadata = fs::metadata(path).ok();
            FileState {
                path: path.clone(),
                size: metadata.as_ref().map(|value| value.len()),
                modified: metadata
                    .and_then(|value| value.modified().ok())
                    .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                    .map(|value| value.as_nanos()),
            }
        })
        .collect()
}

fn parse_args() -> Result<Args, String> {
    let mut values = std::env::args().skip(1);
    let mut root = None;
    let mut input = None;
    let mut output_directory = None;
    let mut max_pdf_bytes = None;
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--root" => root = Some(PathBuf::from(value)),
            "--input" => input = Some(PathBuf::from(value)),
            "--output-directory" => output_directory = Some(PathBuf::from(value)),
            "--max-pdf-bytes" => {
                max_pdf_bytes = Some(value.parse().map_err(|_| "invalid PDF limit")?)
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    let root =
        fs::canonicalize(root.ok_or("missing --root")?).map_err(|error| error.to_string())?;
    let input =
        fs::canonicalize(input.ok_or("missing --input")?).map_err(|error| error.to_string())?;
    if !input.starts_with(&root) {
        return Err("input escapes project root".into());
    }
    let output_directory = output_directory.ok_or("missing --output-directory")?;
    fs::create_dir_all(&output_directory).map_err(|error| error.to_string())?;
    Ok(Args {
        root,
        input,
        output_directory,
        max_pdf_bytes: max_pdf_bytes.ok_or("missing --max-pdf-bytes")?,
    })
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            let _ = emit(json!({ "type": "fatal", "engine": ENGINE, "message": error }));
            std::process::exit(2);
        }
    };
    let mut fonts = FontStore::new();
    fonts.extend(fonts::embedded());
    fonts.extend(fonts::system());
    let fonts = Arc::new(fonts);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            match line {
                Ok(line) => match serde_json::from_str(&line) {
                    Ok(request) => {
                        if tx.send(Input::Request(request)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        if tx.send(Input::Invalid(error.to_string())).is_err() {
                            return;
                        }
                    }
                },
                Err(error) => {
                    let _ = tx.send(Input::Invalid(error.to_string()));
                    break;
                }
            }
        }
        let _ = tx.send(Input::Closed);
    });

    let (mut snapshot, mut dependencies) = compile(&args, Arc::clone(&fonts), None);
    if dependencies.is_empty() {
        dependencies.push(args.input.clone());
    }
    let mut states = file_states(&dependencies);
    loop {
        match rx.recv_timeout(StdDuration::from_millis(75)) {
            Ok(Input::Request(request)) => resolve_click(snapshot.as_ref(), request),
            Ok(Input::Invalid(message)) => {
                let _ = emit(json!({
                    "type": "protocol-error",
                    "engine": ENGINE,
                    "message": message,
                }));
            }
            Ok(Input::Closed) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let next_states = file_states(&dependencies);
                if next_states != states {
                    (snapshot, dependencies) = compile(&args, Arc::clone(&fonts), snapshot);
                    if dependencies.is_empty() {
                        dependencies.push(args.input.clone());
                    }
                    states = file_states(&dependencies);
                }
            }
        }
    }
}
