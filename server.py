"""GitHub Project Radar MCP server.

README
======
1. Install dependencies: ``python -m pip install -r requirements.txt``.
2. Generate a Fernet key with
   ``python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"``
   and configure it as ``APP_ENCRYPTION_KEY`` in Streamlit secrets or the
   process environment. Keep it outside the project and back it up securely:
   losing it makes saved API keys unreadable.
3. Run the dashboard with ``streamlit run app.py``. Create a local account,
   then save a read-only GitHub token and Gemini API key in the sidebar; they
   are encrypted in the local SQLite database.
4. For standalone use, set ``GITHUB_TOKEN`` in the environment. The MCP server
   communicates over stdio and is intended to be launched by an MCP client.
"""

import json
import os
import re

import httpx
from pydantic import BaseModel
from rich.console import Console
from mcp.server.fastmcp import FastMCP

GITHUB_API_URL = "https://api.github.com"
REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_LIMIT = 100

mcp = FastMCP("GitHub-Project-Radar")


class IssueSummary(BaseModel):
    title: str
    body: str
    state: str
    labels: list[str]
    assignee: str | None


class PullRequestSummary(BaseModel):
    title: str
    author: str
    status: str


def _validate_request(repo_name: str, limit: int) -> str | None:
    if not REPOSITORY_PATTERN.fullmatch(repo_name):
        return "Error: repo_name must be in owner/repo format."
    if not 1 <= limit <= MAX_LIMIT:
        return f"Error: limit must be between 1 and {MAX_LIMIT}."
    if not os.environ.get("GITHUB_TOKEN", "").strip():
        return "Error: GITHUB_TOKEN is not set. Configure it before querying GitHub."
    return None


async def _fetch_github(
    url: str, params: dict[str, str | int] | None = None
) -> tuple[object | None, str | None]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return None, "Error: GITHUB_TOKEN is not set. Configure it before querying GitHub."

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "GitHub-Project-Radar",
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0), headers=headers, follow_redirects=True
        ) as client:
            response = await client.get(url, params=params)
    except httpx.TimeoutException:
        return None, "Error: the GitHub API request timed out. Please try again."
    except httpx.RequestError as exc:
        return None, f"Error: could not reach the GitHub API ({exc})."

    if response.status_code == 401:
        return None, "Error: GitHub rejected GITHUB_TOKEN (401 Unauthorized). Check the token."
    if response.status_code == 404:
        return None, "Error: repository not found (404). Check the owner/repo name and access."
    if response.status_code == 403:
        return None, "Error: GitHub denied the request (403). Check token permissions or API rate limits."
    if not response.is_success:
        return None, f"Error: GitHub API returned HTTP {response.status_code}."
    try:
        return response.json(), None
    except ValueError:
        return None, "Error: GitHub returned an invalid JSON response."


@mcp.tool()
async def get_open_issues(
    repo_name: str,
    limit: int = 10,
) -> str:
    """Get recent open issues, excluding pull requests, for an owner/repo."""
    validation_error = _validate_request(repo_name, limit)
    if validation_error:
        return validation_error

    url = f"{GITHUB_API_URL}/repos/{repo_name}/issues"
    issues: list[IssueSummary] = []
    page = 1
    while len(issues) < limit:
        payload, error = await _fetch_github(
            url,
            {"state": "open", "sort": "created", "direction": "desc", "per_page": 100, "page": page},
        )
        if error:
            return error
        if not isinstance(payload, list):
            return "Error: GitHub returned an unexpected issues response."

        for item in payload:
            if not isinstance(item, dict) or "pull_request" in item:
                continue
            labels = item.get("labels", [])
            assignee = item.get("assignee")
            issues.append(
                IssueSummary(
                    title=item.get("title") or "",
                    body=item.get("body") or "",
                    state=item.get("state") or "open",
                    labels=[
                        label.get("name", "")
                        for label in labels
                        if isinstance(label, dict) and label.get("name")
                    ]
                    if isinstance(labels, list)
                    else [],
                    assignee=assignee.get("login")
                    if isinstance(assignee, dict)
                    else None,
                )
            )
            if len(issues) == limit:
                break

        if len(payload) < 100:
            break
        page += 1

    return json.dumps([issue.model_dump() for issue in issues], ensure_ascii=False)


@mcp.tool()
async def get_recent_prs(
    repo_name: str,
    limit: int = 5,
) -> str:
    """Get recent active pull requests, including their title, author, and status."""
    validation_error = _validate_request(repo_name, limit)
    if validation_error:
        return validation_error

    payload, error = await _fetch_github(
        f"{GITHUB_API_URL}/repos/{repo_name}/pulls",
        {"state": "open", "sort": "updated", "direction": "desc", "per_page": limit},
    )
    if error:
        return error
    if not isinstance(payload, list):
        return "Error: GitHub returned an unexpected pull requests response."

    pull_requests = [
        PullRequestSummary(
            title=item.get("title") or "",
            author=(item.get("user") or {}).get("login", "unknown")
            if isinstance(item.get("user"), dict)
            else "unknown",
            status="draft" if item.get("draft") else "open",
        )
        for item in payload
        if isinstance(item, dict)
    ]
    return json.dumps([pull_request.model_dump() for pull_request in pull_requests], ensure_ascii=False)


if __name__ == "__main__":
    Console(stderr=True).print("[bold green]Starting GitHub-Project-Radar MCP server[/]")
    mcp.run(transport="stdio")
