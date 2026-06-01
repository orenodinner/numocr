use std::path::PathBuf;
use std::process::Command;

pub fn new_tesseract_command() -> Command {
    Command::new(tesseract_executable())
}

fn tesseract_executable() -> PathBuf {
    if let Ok(path) = std::env::var("TESSERACT_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            return path;
        }
    }

    #[cfg(target_os = "windows")]
    {
        let common_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ];
        for path in common_paths {
            let path = PathBuf::from(path);
            if path.exists() {
                return path;
            }
        }
    }

    PathBuf::from("tesseract")
}
