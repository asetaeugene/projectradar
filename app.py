from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import bcrypt
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken
from google import genai
from google.genai.errors import APIError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult

MODEL_FALLBACK_CHAIN = (
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
)
DATABASE_PATH = Path(__file__).with_name("app_credentials.sqlite3")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,64}$")
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _encryption_key() -> bytes:
    try:
        key = st.secrets.get("APP_ENCRYPTION_KEY")
    except (FileNotFoundError, KeyError):
        key = None
    key = key or os.environ.get("APP_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError(
            "Credential encryption is not configured. Set APP_ENCRYPTION_KEY "
            "in Streamlit secrets or the process environment."
        )
    try:
        Fernet(key.encode("ascii"))
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise RuntimeError(
            "APP_ENCRYPTION_KEY must be a valid Fernet key."
        ) from exc
    return key.encode("ascii")


def _initialize_database() -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash BLOB NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS api_credentials (
                username TEXT PRIMARY KEY REFERENCES users(username),
                gemini_ciphertext BLOB,
                github_ciphertext BLOB
            )
            """
        )


def _account_count() -> int:
    with sqlite3.connect(DATABASE_PATH) as connection:
        return connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def _create_account(username: str, password: str) -> None:
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )


def _authenticate(username: str, password: str) -> bool:
    if len(password.encode("utf-8")) > 72:
        return False
    with sqlite3.connect(DATABASE_PATH) as connection:
        record = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
    return bool(
        record
        and bcrypt.checkpw(password.encode("utf-8"), bytes(record[0]))
    )


def _save_credentials(
    username: str, gemini_key: str, github_token: str
) -> None:
    cipher = Fernet(_encryption_key())
    with sqlite3.connect(DATABASE_PATH) as connection:
        existing = connection.execute(
            """
            SELECT gemini_ciphertext, github_ciphertext
            FROM api_credentials WHERE username = ?
            """,
            (username,),
        ).fetchone()
        current_gemini = existing[0] if existing else None
        current_github = existing[1] if existing else None
        encrypted_gemini = (
            cipher.encrypt(gemini_key.encode("utf-8"))
            if gemini_key
            else current_gemini
        )
        encrypted_github = (
            cipher.encrypt(github_token.encode("utf-8"))
            if github_token
            else current_github
        )
        connection.execute(
            """
            INSERT INTO api_credentials
                (username, gemini_ciphertext, github_ciphertext)
            VALUES (?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                gemini_ciphertext = excluded.gemini_ciphertext,
                github_ciphertext = excluded.github_ciphertext
            """,
            (username, encrypted_gemini, encrypted_github),
        )


def _load_credentials(username: str) -> tuple[str, str]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        record = connection.execute(
            """
            SELECT gemini_ciphertext, github_ciphertext
            FROM api_credentials WHERE username = ?
            """,
            (username,),
        ).fetchone()
    if not record or not record[0] or not record[1]:
        raise RuntimeError(
            "Save both a Gemini API key and a GitHub token in Settings before analyzing."
        )
    cipher = Fernet(_encryption_key())
    try:
        return (
            cipher.decrypt(bytes(record[0])).decode("utf-8"),
            cipher.decrypt(bytes(record[1])).decode("utf-8"),
        )
    except (InvalidToken, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "Saved credentials could not be decrypted. Check that APP_ENCRYPTION_KEY "
            "matches the key used when they were saved."
        ) from exc


def _tool_result_text(result: CallToolResult) -> str:
    text_parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text" and isinstance(item.text, str)
    ]
    content = "\n".join(text_parts)
    if result.isError:
        return f"Tool error: {content or 'The MCP tool returned an error.'}"
    return content


def _result_count(content: str) -> int | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return len(parsed) if isinstance(parsed, list) else None


def _exception_details(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        details = [_exception_details(inner) for inner in error.exceptions]
        return "; ".join(detail for detail in details if detail)
    return str(error).strip() or type(error).__name__


def _is_capacity_error(error: APIError) -> bool:
    code = str(getattr(error, "code", "")).upper()
    status = str(getattr(error, "status", "")).upper()
    message = str(error).upper()
    return (
        code == "503"
        or status in {"503", "UNAVAILABLE"}
        or "503" in message
        or "UNAVAILABLE" in message
    )


async def _analyze_project(
    repo_name: str,
    github_token: str,
    active_gemini_key: str,
    status_container,
) -> tuple[str, dict[str, int | None], dict[str, object]]:
    server_path = Path(__file__).with_name("server.py")
    child_env = os.environ.copy()
    child_env["GITHUB_TOKEN"] = github_token
    child_env["GEMINI_API_KEY"] = active_gemini_key
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=child_env,
    )
    metrics: dict[str, int | None] = {"issues": None, "pull_requests": None}
    raw_tool_payloads: dict[str, object] = {}

    status_container.write("Connecting to FastMCP Server...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            mcp_tool_names = {tool.name for tool in mcp_tools}
            requested_tools = ("get_open_issues", "get_recent_prs")
            unavailable = [name for name in requested_tools if name not in mcp_tool_names]
            if unavailable:
                raise RuntimeError(
                    f"MCP server is missing required tools: {', '.join(unavailable)}."
                )

            status_container.write("Fetching open issues & pull requests...")
            github_data: dict[str, object] = {"repository": repo_name}
            for tool_name in requested_tools:
                result = await session.call_tool(
                    tool_name, arguments={"repo_name": repo_name}
                )
                tool_output = _tool_result_text(result)
                if result.isError or tool_output.startswith(("Error:", "Tool error:")):
                    raise RuntimeError(tool_output)

                try:
                    parsed_output = json.loads(tool_output)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"MCP tool {tool_name} returned invalid JSON."
                    ) from exc
                raw_tool_payloads[tool_name] = parsed_output
                count = _result_count(tool_output)
                if tool_name == "get_open_issues":
                    metrics["issues"] = count
                    github_data["open_issues"] = parsed_output
                else:
                    metrics["pull_requests"] = count
                    github_data["open_pull_requests"] = parsed_output

    prompt = (
        "Prepare a concise, structured project briefing in Markdown from the GitHub "
        "data below. Include an overview, open issues, and pull requests. Summarize "
        "notable themes and status using only the supplied data; do not invent facts. "
        "Explicitly note when either list is empty.\n\n"
        f"GitHub data:\n{json.dumps(github_data, ensure_ascii=False, indent=2)}"
    )
    status_container.write("Synthesizing health report via Gemini...")
    client = genai.Client(api_key=active_gemini_key)
    try:
        response = None
        for index, model in enumerate(MODEL_FALLBACK_CHAIN):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt,
                )
                break
            except APIError as exc:
                if not _is_capacity_error(exc):
                    raise
                if index == len(MODEL_FALLBACK_CHAIN) - 1:
                    raise
                next_model = MODEL_FALLBACK_CHAIN[index + 1]
                warning = (
                    f"Gemini model {model} is unavailable or at capacity; "
                    f"trying {next_model}."
                )
                print(f"Warning: {warning}", file=sys.stderr)
                status_container.write(f"Warning: {warning}")
    finally:
        client.close()

    if response is None:
        raise RuntimeError("Gemini did not return a response.")
    briefing = (response.text or "").strip()
    if not briefing:
        raise RuntimeError("Gemini returned no text for the project briefing.")
    return briefing, metrics, raw_tool_payloads


def _preset_repository(repository: str) -> None:
    st.session_state["repo_name"] = repository


st.set_page_config(page_title="GitHub Project Radar", page_icon="📡", layout="wide")
_initialize_database()

if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None

if not st.session_state["authenticated_user"]:
    st.title("GitHub Project Radar")
    st.caption("Sign in to securely manage credentials and generate project briefings.")
    try:
        if _account_count() == 0:
            st.subheader("Create the first account")
            st.info("This local account protects access to the app on this deployment.")
            with st.form("create_account_form", clear_on_submit=True):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                create_account = st.form_submit_button(
                    "Create account", type="primary"
                )
            if create_account:
                if not USERNAME_PATTERN.fullmatch(new_username.strip()):
                    st.error(
                        "Username must be 3-64 characters and contain only letters, "
                        "numbers, dots, underscores, or hyphens."
                    )
                elif len(new_password) < 12:
                    st.error("Choose a password with at least 12 characters.")
                elif len(new_password.encode("utf-8")) > 72:
                    st.error("Password must be no longer than 72 UTF-8 bytes.")
                elif new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    _create_account(new_username.strip(), new_password)
                    st.session_state["authenticated_user"] = new_username.strip()
                    st.rerun()
        else:
            st.subheader("Log in")
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login = st.form_submit_button("Log in", type="primary")
            if login:
                if _authenticate(username.strip(), password):
                    st.session_state["authenticated_user"] = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    except sqlite3.Error as exc:
        st.error(f"Authentication storage failed: {_exception_details(exc)}")
    st.stop()

username = st.session_state["authenticated_user"]
st.title("GitHub Project Radar")
st.caption(
    "A Gemini-powered project briefing built from live GitHub issues and pull requests."
)

with st.sidebar:
    st.header("Settings")
    st.caption(f"Signed in as {username}")
    with st.form("api_credentials_form", clear_on_submit=True):
        settings_gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            help="Saved encrypted locally. Leave blank to keep the saved key.",
        )
        settings_github_token = st.text_input(
            "GitHub token",
            type="password",
            help="Use a read-only token. Saved encrypted locally.",
        )
        save_credentials = st.form_submit_button("Save credentials")
    if save_credentials:
        if not settings_gemini_key and not settings_github_token:
            st.info("Enter at least one key to save or update.")
        else:
            try:
                _save_credentials(
                    username, settings_gemini_key, settings_github_token
                )
                st.success("Credentials saved securely.")
            except (RuntimeError, sqlite3.Error) as exc:
                st.error(f"Could not save credentials: {_exception_details(exc)}")
    if st.button("Log out", use_container_width=True):
        st.session_state["authenticated_user"] = None
        for key in ("briefing", "project_metrics", "raw_tool_payloads"):
            st.session_state.pop(key, None)
        st.rerun()

st.subheader("Analyze a repository")
repo_name = st.text_input(
    "Repository",
    placeholder="owner/repo",
    key="repo_name",
)
preset_columns = st.columns(3)
presets = ("pallets/flask", "streamlit/streamlit", "fastapi/fastapi")
for column, repository in zip(preset_columns, presets):
    column.button(
        repository,
        on_click=_preset_repository,
        args=(repository,),
        use_container_width=True,
    )
analyze = st.button("Analyze Project", type="primary")

if analyze:
    if not REPOSITORY_PATTERN.fullmatch(repo_name.strip()):
        st.error("Enter a valid repository in owner/repo format.")
    else:
        try:
            gemini_api_key, github_token = _load_credentials(username)
            with st.status("Analyzing Repository...", expanded=True) as status:
                try:
                    briefing, project_metrics, raw_tool_payloads = asyncio.run(
                        _analyze_project(
                            repo_name.strip(),
                            github_token,
                            gemini_api_key,
                            status,
                        )
                    )
                    st.session_state["briefing"] = briefing
                    st.session_state["project_metrics"] = project_metrics
                    st.session_state["raw_tool_payloads"] = raw_tool_payloads
                    status.update(label="Repository analysis complete.", state="complete")
                except (
                    APIError,
                    McpError,
                    OSError,
                    RuntimeError,
                    ValueError,
                    sqlite3.Error,
                    ExceptionGroup,
                ):
                    status.update(label="Repository analysis failed.", state="error")
                    raise
        except (
            APIError,
            McpError,
            OSError,
            RuntimeError,
            ValueError,
            sqlite3.Error,
            ExceptionGroup,
        ) as exc:
            st.error(f"Project analysis failed: {_exception_details(exc)}")

if "project_metrics" in st.session_state:
    metrics = st.session_state["project_metrics"]
    issue_metric, pr_metric = st.columns(2)
    issue_metric.metric(
        "Open issues",
        metrics["issues"] if metrics["issues"] is not None else "Unavailable",
    )
    pr_metric.metric(
        "Open pull requests",
        metrics["pull_requests"]
        if metrics["pull_requests"] is not None
        else "Unavailable",
    )

if "briefing" in st.session_state:
    st.markdown(st.session_state["briefing"])
    st.download_button(
        "Download executive summary",
        data=st.session_state["briefing"],
        file_name="repo_briefing.md",
        mime="text/markdown",
    )

with st.expander("🔍 Inspect MCP Tool Payloads"):
    if "raw_tool_payloads" in st.session_state:
        st.json(st.session_state["raw_tool_payloads"])
    else:
        st.caption("Run an analysis to inspect the GitHub MCP tool payloads.")
