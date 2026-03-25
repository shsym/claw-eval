"""Sandbox tool definitions for containerised agent execution."""

from __future__ import annotations

from ..models.tool import ToolSpec

# ---------------------------------------------------------------------------
# Tool specifications
# ---------------------------------------------------------------------------

_SHELL_EXEC = ToolSpec(
    name="sandbox_shell_exec",
    description="Execute a shell command inside the sandbox and return stdout/stderr.",
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max seconds before the command is killed (default: 30).",
                "default": 30,
            },
        },
        "required": ["command"],
    },
)

_FILE_READ = ToolSpec(
    name="sandbox_file_read",
    description="Read the contents of a file at the given path.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
        },
        "required": ["path"],
    },
)

_FILE_WRITE = ToolSpec(
    name="sandbox_file_write",
    description="Write content to a file at the given path (creates parent directories if needed).",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
)

_BROWSER_SCREENSHOT = ToolSpec(
    name="sandbox_browser_screenshot",
    description=(
        "Capture screenshots of a web page over time. "
        "Opens the URL in a headless browser, then takes multiple screenshots "
        "at regular intervals to show animation progress. "
        "Use this to preview and verify your generated web pages and animations."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to navigate to and capture.",
            },
            "wait_seconds": {
                "type": "number",
                "description": "Total observation time in seconds (default: 2.0).",
                "default": 2.0,
            },
            "frame_count": {
                "type": "integer",
                "description": "Number of screenshots to capture over the observation period (default: 4).",
                "default": 4,
            },
        },
        "required": ["url"],
    },
)

_READ_MEDIA = ToolSpec(
    name="sandbox_read_media",
    description=(
        "Read and preview a media file (image, video, or PDF). "
        "For video: extracts frames at specified intervals. "
        "For PDF: renders pages as images. "
        "Returns metadata and base64-encoded frame images."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the media file.",
            },
            "media_type": {
                "type": "string",
                "description": "Type of media: 'auto', 'image', 'video', or 'pdf'. Default: auto-detect from extension.",
                "default": "auto",
            },
            "max_frames": {
                "type": "integer",
                "description": "Maximum number of frames to extract from video (default: 8).",
                "default": 8,
            },
            "fps": {
                "type": "number",
                "description": "Frames per second to extract from video (default: 1.0).",
                "default": 1.0,
            },
            "start_time": {
                "type": "number",
                "description": "Start time in seconds for video extraction (default: 0.0).",
                "default": 0.0,
            },
            "end_time": {
                "type": "number",
                "description": "End time in seconds for video extraction. None means end of video.",
            },
            "screen_size": {
                "type": "string",
                "description": "Resize output to this dimension, e.g. '1280x720'.",
            },
            "pdf_pages": {
                "type": "string",
                "description": "Pages to render for PDF: 'all', '1-3', or '1,3,5' (default: 'all').",
                "default": "all",
            },
            "dpi": {
                "type": "integer",
                "description": "DPI for PDF rendering (default: 150).",
                "default": 150,
            },
        },
        "required": ["path"],
    },
)

_PDF2IMAGE = ToolSpec(
    name="sandbox_pdf2image",
    description="Render PDF pages as images. Simpler interface for document tasks.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the PDF file.",
            },
            "pages": {
                "type": "string",
                "description": "Pages to render: 'all', '1-3', or '1,3,5' (default: 'all').",
                "default": "all",
            },
            "dpi": {
                "type": "integer",
                "description": "DPI for rendering (default: 150).",
                "default": 150,
            },
            "max_dimension": {
                "type": "integer",
                "description": "Max pixel dimension for output images (default: 1024).",
                "default": 1024,
            },
        },
        "required": ["path"],
    },
)

_FILE_DOWNLOAD = ToolSpec(
    name="sandbox_file_download",
    description="Download a file as binary (base64-encoded). Use for retrieving generated files (mp4, gif, html, etc.).",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to download.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum file size in bytes (default: 50MB).",
                "default": 50000000,
            },
        },
        "required": ["path"],
    },
)

# Full list of all sandbox tools
SANDBOX_TOOLS: list[ToolSpec] = [
    _SHELL_EXEC,
    _FILE_READ,
    _FILE_WRITE,
    _BROWSER_SCREENSHOT,
    _READ_MEDIA,
    _PDF2IMAGE,
    _FILE_DOWNLOAD,
]


def get_sandbox_tools(
    *,
    enable_shell: bool = True,
    enable_browser: bool = True,
    enable_file: bool = True,
    enable_media: bool = True,
) -> list[ToolSpec]:
    """Return a filtered list of sandbox tools based on capability flags."""
    tools: list[ToolSpec] = []
    if enable_shell:
        tools.append(_SHELL_EXEC)
    if enable_file:
        tools.append(_FILE_READ)
        tools.append(_FILE_WRITE)
    if enable_browser:
        tools.append(_BROWSER_SCREENSHOT)
    if enable_media:
        tools.append(_READ_MEDIA)
        tools.append(_PDF2IMAGE)
        tools.append(_FILE_DOWNLOAD)
    return tools
